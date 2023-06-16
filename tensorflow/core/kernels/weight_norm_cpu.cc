#include <iostream>
#include <cmath>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

template<class T>
class WeightNormOp : public OpKernel {
public:
    explicit WeightNormOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &data_type));
        assert(data_type == (DT_FLOAT || DT_DOUBLE));
    }

    ~WeightNormOp() {
    }

    void Compute(OpKernelContext* ctx) override {        
        // Get weight tensors
        const Tensor& init_weight_tensor = ctx->input(0);
        TensorShape shape = init_weight_tensor.shape();
        int64_t input_dims = init_weight_tensor.dims();
        OP_REQUIRES(ctx, (input_dims == 2),
                    errors::InvalidArgument("Init weights is not 2D, currently we only support 2D."));
        int64_t row = init_weight_tensor.dim_size(0);
        int64_t col = init_weight_tensor.dim_size(1);

        const Tensor& trainable_weight_tensor = ctx->input(1);
        TensorShape trainable_shape = trainable_weight_tensor.shape();
        int64_t trainable_input_dims = trainable_weight_tensor.dims();
        OP_REQUIRES(ctx, (trainable_input_dims == 2),
                    errors::InvalidArgument("Trainable weights is not 2D, currently we only support 2D."));
        int64_t trainable_row = trainable_weight_tensor.dim_size(0);
        int64_t trainable_col = trainable_weight_tensor.dim_size(1);
        OP_REQUIRES(ctx, (trainable_row == row && trainable_col == col),
                    errors::InvalidArgument("Trainable weights should have same shape with init weights. Pls check use case."));

        // Get weight_norm dim, for a 2D tensor, norm_dim should be either 0/1/2
        // 0-row / 1-col / 2-tuple(0,1)
        const Tensor& norm_dim_tensor = ctx->input(2);
        // Dim should be a scalar integer
        OP_REQUIRES(ctx,
                (TensorShapeUtils::IsScalar(norm_dim_tensor.shape()) ||
                 (TensorShapeUtils::IsVector(norm_dim_tensor.shape()) &&
                  norm_dim_tensor.shape().dim_size(0) == 1)),
                errors::InvalidArgument(
                    "Norm dim tensor should be a scalar integer, but got shape ",
                    norm_dim_tensor.shape().DebugString()));
        int64_t norm_dim = internal::SubtleMustCopy(norm_dim_tensor.scalar<int64_t>()());
        //int64_t axis = norm_dim < 0 ? input_dims : norm_dim;
        int64_t axis = norm_dim;
        OP_REQUIRES(ctx, (0 <= axis && axis <= input_dims),
                    errors::InvalidArgument(
                        "WeightNormOp : Expected norm dimensions in the range "
                        "[0, ", input_dims, "], but got ", axis));

        // Handle input tensors
        auto src_data = init_weight_tensor.flat<T>();
        auto trainable_src_data = trainable_weight_tensor.flat<T>();
        
        // Allocate for output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output_tensor));
        auto output = output_tensor->flat<T>();

        const auto num_threads =
            ctx->device()->tensorflow_cpu_worker_threads()->num_threads;

        // Small constant to avoid division by zero
        const Tensor& l2norm_epsilon = ctx->input(3);
        auto epsilon_flat = l2norm_epsilon.flat<T>();
        T eps = epsilon_flat(0);

        const int64_t cost_per_row = col * sizeof(T);
        const int64_t cost_per_col = row * sizeof(T);
        // Case1. norm is calculated over the entire array
        if (axis == input_dims){
            auto scaling_factor = 0.0;
            auto tmp_sum = 0.0;
            auto shard_fn = [&](int64 start, int64 limit) {
                for (int64 i = start; i < limit; i++) {
                    for (int j = 0; j < col; j++) {
                        tmp_sum += pow(src_data(i * col + j), 2);
                    }
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, row, cost_per_row, shard_fn);
            scaling_factor = sqrt(tmp_sum) + eps;
            auto inverse_scale = 1 / scaling_factor;
            auto shard_fn2 = [&](int64 start, int64 limit) {
                for (int64 i = start; i < limit; i++) {
                    for (int j = 0; j < col; j++) {
                        output(i * col + j) = src_data(i * col + j) * inverse_scale * trainable_src_data(i * col + j);
                    }
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, row, cost_per_row, shard_fn2);
        } else if (axis == 0){
            // Case2. norm is calculated along axis 0 (per row)
            std::vector<T> scaling_factors(col);
            std::vector<T> inverse_scale(col);
            auto shard_fn = [&](int64 start, int64 limit) {
                for (int i = start; i < limit; i++) {
                    auto tmp_sum = 0.0;
                    for (int j = 0; j < row; j++){
                        tmp_sum += pow(src_data(j * col + i), 2);
                    }
                    scaling_factors[i] = sqrt(tmp_sum) + eps;
                    inverse_scale[i] = 1 / scaling_factors[i];
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, col, cost_per_col, shard_fn);
            // Here use another shard func to calculate output by row for better
            // memory access locality, which out-performs than by col
            auto shard_fn2 = [&](int64 start, int64 limit) {
                for (int64 i = start; i < limit; i++) {
                    for (int j = 0; j < col; j++){
                        output(i * col + j) = src_data(i * col + j) * inverse_scale[j] * trainable_src_data(i * col + j);
                    }
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, row, cost_per_row, shard_fn2);
        } else {
            // Case3. norm is calculated along axis 1 (per col)
            std::vector<T> scaling_factors(row);
            std::vector<T> inverse_scale(row);
            auto shard_fn = [&](int64 start, int64 limit) {
                for (int i = start; i < limit; i++) {
                    auto tmp_sum = 0.0;
                    for (int j = 0; j < col; j++){
                        tmp_sum += pow(src_data(i * col + j), 2);
                    }
                    scaling_factors[i] = sqrt(tmp_sum) + eps;
                    inverse_scale[i] = 1 / scaling_factors[i];
                    for (int j = 0; j < col; j++){
                        output(i * col + j) = src_data(i * col + j) * inverse_scale[i] * trainable_src_data(i * col + j);
                    }
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, row, cost_per_row, shard_fn);
        }
    }

private:
  void init() {
}

private:
  DataType data_type;
};

REGISTER_OP("WeightNorm")
    .Input("init_weights: T")
    .Input("trainable_weights: T")
    .Input("l2norm_axis: int64")
    .Input("l2norm_epsilon: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);


// Register the WeightNormOp kernel
REGISTER_KERNEL_BUILDER(Name("WeightNorm")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        WeightNormOp<float>);
REGISTER_KERNEL_BUILDER(Name("WeightNorm")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        WeightNormOp<double>);

}  // namespace tensorflow