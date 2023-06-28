#include <iostream>
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
        OP_REQUIRES(ctx, (trainable_input_dims == 1),
                    errors::InvalidArgument("Got trainable weights from a tf.norm along axis=0 without setting keepDim=True, thus trainable weights should be 1D"));
        //int64_t trainable_row = trainable_weight_tensor.dim_size(0);
        int64_t trainable_col = trainable_weight_tensor.dim_size(0);
        OP_REQUIRES(ctx, trainable_col == col,
                    errors::InvalidArgument("Trainable weights should have same column with init weights. Pls check use case."));

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
                        tmp_sum += src_data(i * col + j) * src_data(i * col + j);
                    }
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, row, cost_per_row, shard_fn);
            scaling_factor = sqrt(tmp_sum) + eps;
            auto inverse_scale = 1 / scaling_factor;
            auto shard_fn2 = [&](int64 start, int64 limit) {
                for (int64 i = start; i < limit; i++) {
                    for (int j = 0; j < col; j++) {
                        output(i * col + j) = src_data(i * col + j) * trainable_src_data(j) * inverse_scale;
                    }
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, row, cost_per_row, shard_fn2);
        } else if (axis == 0){
            // Case2. norm is calculated along axis 0 (per row)
            std::vector<T> scaling_factors(col, 0.0);
            std::vector<T> inverse_scale(col, 0.0);

            // row be the outer loop to make src_data continuous
            auto shard_fn = [&](int64 start, int64 limit) {
                for (int j = 0; j < row; j++){
                    for (int i = start; i < limit; i++) {
                        scaling_factors[i] += src_data(j * col + i) * src_data(j * col + i);
                    }
                }
                for (int i = start; i < limit; i++) {
                    scaling_factors[i] = sqrt(scaling_factors[i]) + eps;
                    inverse_scale[i] = 1 / scaling_factors[i];
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, col, cost_per_col, shard_fn);
            // Here use another shard func to calculate output by row for better
            // memory access locality, which out-performs than by col
            auto shard_fn2 = [&](int64 start, int64 limit) {
                for (int64 i = start; i < limit; i++) {
                    for (int j = 0; j < col; j++){
                        output(i * col + j) = src_data(i * col + j) * inverse_scale[j] * trainable_src_data(j);
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
                    auto tmp_square_sum = 0.0;
                    for (int j = 0; j < col; j++){
                        tmp_square_sum += src_data(i * col + j) * src_data(i * col + j);
                    }
                    scaling_factors[i] = sqrt(tmp_square_sum) + eps;
                    inverse_scale[i] = 1 / scaling_factors[i];
                    for (int j = 0; j < col; j++){
                        output(i * col + j) = src_data(i * col + j) * inverse_scale[i] * trainable_src_data(j);
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

/* l2_norm for non-complex is calculated as
        mul(x, rsqrt(reduce_sum(square(x), axis)))
   the gradient with respect to x should be like below formula
        rsqrt(reduce_sum(square(x), axis)) + reduce_sum(x, axis) * (-(reduce_sum(square(x), axis))^(-3/2) * x)
   which equals to
        rsqrt(reduce_sum(square(x), axis)) + reduce_sum(x, axis) * (-(rsqrt(reduce_sum(square(x), axis))^3 * x)

   weight_norm for non-complex is calculated as
        mul(l2_norm(x), y)
   where x(init_weight) and l2_norm(x) have shape (row, col) and y(trainable_weight) has shape (col, )
   the multiplication should reduce l2_norm(x) along axis=0
   the gradient with respect to y should be like below formula
        reduce_sum(x, axis=0) * rsqrt(reduce_sum(square(x), axis))
*/
template<class T>
class WeightNormGradOp : public OpKernel {
public:
    explicit WeightNormGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &data_type));
        assert(data_type == (DT_FLOAT || DT_DOUBLE));
    }

    ~WeightNormGradOp() {
    }

    void Compute(OpKernelContext* ctx) override {
        // Get backward gradient
        const Tensor& grad_tensor = ctx->input(0);
        // Get weight tensors
        const Tensor& init_weight_tensor = ctx->input(1);
        TensorShape shape = init_weight_tensor.shape();
        int64_t input_dims = init_weight_tensor.dims();
        OP_REQUIRES(ctx, (input_dims == 2),
                    errors::InvalidArgument("Init weights is not 2D, currently we only support 2D."));
        int64_t row = init_weight_tensor.dim_size(0);
        int64_t col = init_weight_tensor.dim_size(1);

        const Tensor& trainable_weight_tensor = ctx->input(2);
        TensorShape trainable_shape = trainable_weight_tensor.shape();
        int64_t trainable_input_dims = trainable_weight_tensor.dims();
        OP_REQUIRES(ctx, (trainable_input_dims == 1),
                    errors::InvalidArgument("Got trainable weights from a tf.norm along axis=0 without setting keepDim=True, thus trainable weights should be 1D"));
        //int64_t trainable_row = trainable_weight_tensor.dim_size(0);
        int64_t trainable_col = trainable_weight_tensor.dim_size(0);
        OP_REQUIRES(ctx, trainable_col == col,
                    errors::InvalidArgument("Trainable weights should have same column with init weights. Pls check use case."));

        // Get weight_norm dim, for a 2D tensor, norm_dim should be either 0/1/2
        // 0-row / 1-col / 2-tuple(0,1)
        const Tensor& norm_dim_tensor = ctx->input(3);
        // Dim should be a scalar integer
        OP_REQUIRES(ctx,
                (TensorShapeUtils::IsScalar(norm_dim_tensor.shape()) ||
                 (TensorShapeUtils::IsVector(norm_dim_tensor.shape()) &&
                  norm_dim_tensor.shape().dim_size(0) == 1)),
                errors::InvalidArgument(
                    "Norm dim tensor should be a scalar integer, but got shape ",
                    norm_dim_tensor.shape().DebugString()));
        int64_t norm_dim = internal::SubtleMustCopy(norm_dim_tensor.scalar<int64_t>()());
        int64_t axis = norm_dim;
        OP_REQUIRES(ctx, (0 <= axis && axis <= input_dims),
                    errors::InvalidArgument(
                        "WeightNormOp : Expected norm dimensions in the range "
                        "[0, ", input_dims, "], but got ", axis));

        // Handle input tensors
        auto grad_data = grad_tensor.flat<T>();
        auto src_data = init_weight_tensor.flat<T>();
        auto trainable_src_data = trainable_weight_tensor.flat<T>();

        // Allocate for output tensor
        Tensor *output_init_tensor = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output_init_tensor));
        auto output_init = output_init_tensor->flat<T>();
        Tensor *output_trainable_tensor = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, shape, &output_trainable_tensor));
        auto output_trainable = output_trainable_tensor->flat<T>();

        const auto num_threads =
            ctx->device()->tensorflow_cpu_worker_threads()->num_threads;

        // Small constant to avoid division by zero
        const Tensor& l2norm_epsilon = ctx->input(4);
        auto epsilon_flat = l2norm_epsilon.flat<T>();
        T eps = epsilon_flat(0);

        const int64_t cost_per_row = col * sizeof(T);
        const int64_t cost_per_col = row * sizeof(T);

        // Case1. norm is calculated over the entire array
        if (axis == input_dims){
            std::vector<T> sum_factors(col, 0.0);
            auto scaling_factor = 0.0;
            auto tmp_square_sum = 0.0;
            auto shard_fn = [&](int64 start, int64 limit) {
                for (int64 i = start; i < limit; i++) {
                    for (int j = 0; j < col; j++) {
                        sum_factors[j] += src_data(i * col + j);
                        tmp_square_sum += src_data(i * col + j) * src_data(i * col + j);
                    }
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, row, cost_per_row, shard_fn);
            scaling_factor = sqrt(tmp_square_sum) + eps;
            auto inverse_scale = 1 / scaling_factor;
            auto shard_fn2 = [&](int64 start, int64 limit) {
                for (int64 i = start; i < limit; i++) {
                    for (int j = 0; j < col; j++) {
                        //output_init(i * col + j) = (inverse_scale + sum_factors[j] * (- inverse_scale * inverse_scale * inverse_scale * src_data(i * col + j))) * trainable_src_data(j) * grad_data(i * col + j);
                        output_trainable(i * col + j) = sum_factors[j] * inverse_scale * grad_data(i * col + j);
                    }
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, row, cost_per_row, shard_fn2);
        } else if (axis == 0){
            // Case2. norm is calculated along axis 0 (per row)
            std::vector<T> sum_factors(col, 0.0);
            std::vector<T> scaling_factors(col, 0.0);
            std::vector<T> inverse_scale(col, 0.0);

            // row be the outer loop to make src_data continuous
            auto shard_fn = [&](int64 start, int64 limit) {
                for (int j = 0; j < row; j++){
                    for (int i = start; i < limit; i++) {
                        sum_factors[i] += src_data(j * col + i);
                        scaling_factors[i] += src_data(j * col + i) * src_data(j * col + i);
                    }
                }
                for (int i = start; i < limit; i++) {
                    scaling_factors[i] = sqrt(scaling_factors[i]) + eps;
                    inverse_scale[i] = 1 / scaling_factors[i];
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, col, cost_per_col, shard_fn);
            auto shard_fn2 = [&](int64 start, int64 limit) {
                for (int64 i = start; i < limit; i++) {
                    for (int j = 0; j < col; j++){
                        output_init(i * col + j) = (inverse_scale[j] + sum_factors[j] * (- inverse_scale[j] * inverse_scale[j] * inverse_scale[j] * src_data(i * col + j))) * trainable_src_data(j) * grad_data(i * col + j);
                        output_trainable(i * col + j) = sum_factors[j] * inverse_scale[j] * grad_data(i * col + j);
                    }
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, row, cost_per_row, shard_fn2);
        } else{
            // Case3. norm is calculated along axis 1 (per col)
            std::vector<T> sum_factors(col, 0.0);
            std::vector<T> scaling_factors(row);
            std::vector<T> inverse_scale(row);
            auto shard_fn = [&](int64 start, int64 limit) {
                for (int i = start; i < limit; i++) {
                    auto tmp_square_sum = 0.0;
                    for (int j = 0; j < col; j++){
                        tmp_square_sum += src_data(i * col + j) * src_data(i * col + j);
                    }
                    scaling_factors[i] = sqrt(tmp_square_sum) + eps;
                    inverse_scale[i] = 1 / scaling_factors[i];
                    for (int j = 0; j < col; j++){
                        sum_factors[j] += src_data(i * col + j) * inverse_scale[i];
                        //output_init(i * col + j) = (inverse_scale[i] + sum_factors[j] * (- inverse_scale[i] * inverse_scale[i] * inverse_scale[i] * src_data(i * col + j))) * trainable_src_data(j) * grad_data(i * col + j);
                    }
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, row, cost_per_row, shard_fn);
            auto shard_fn2 = [&](int64 start, int64 limit) {
                for (int i = start; i < limit; i++) {
                    for (int j = 0; j < col; j++){
                        output_trainable(i * col + j) = sum_factors[j] * grad_data(i * col + j);
                    }
                }
            };
            Shard(num_threads, ctx->device()->tensorflow_cpu_worker_threads()->workers, row, cost_per_row, shard_fn2);
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

REGISTER_OP("WeightNormGrad")
    .Input("grad: T")
    .Input("init_weights: T")
    .Input("trainable_weights: T")
    .Input("l2norm_axis: int64")
    .Input("l2norm_epsilon: T")
    .Output("output_init: T")
    .Output("output_trainable: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// Register the WeightNorm/WeightNormGrad kernel
REGISTER_KERNEL_BUILDER(Name("WeightNorm")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        WeightNormOp<float>);
REGISTER_KERNEL_BUILDER(Name("WeightNorm")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        WeightNormOp<double>);
REGISTER_KERNEL_BUILDER(Name("WeightNormGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        WeightNormGradOp<float>);
REGISTER_KERNEL_BUILDER(Name("WeightNormGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        WeightNormGradOp<double>);

}  // namespace tensorflow