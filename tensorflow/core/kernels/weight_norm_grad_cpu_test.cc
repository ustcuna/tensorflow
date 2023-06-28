#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

enum class Device { CPU };

class WeightNormGradTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, DataType data_type) {
    // weight_norm_grad op.
    TF_CHECK_OK(NodeDefBuilder("weight_norm_grad", "WeightNormGrad")
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_INT64))
                  .Input(FakeInput(DT_FLOAT))
                  .Attr("T", data_type)
                  .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

//----------------------------------------------------------------------------//
//                            Accuracy Check                                  //
//----------------------------------------------------------------------------//
TEST_F(WeightNormGradTest, Norm_Test_All_Dim) {
  MakeOpAndSetDevice(Device::CPU, DT_FLOAT);
  // Set up the input tensor.
  AddInputFromArray<float>(TensorShape({3, 5}), {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0});
  AddInputFromArray<float>(TensorShape({3, 5}), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  AddInputFromArray<float>(TensorShape({5, }), {5.0, 4.0, 3.0, 2.0, 1.0});
  AddInput<int64_t>(TensorShape({}), [](int i) { return 2; });
  AddInput<float>(TensorShape({}), [](float i) { return 1e-6; });
  
  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    // Set up the expected output tensor.
    Tensor expected_init_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    Tensor expected_trainable_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    float output_array[15];
    // correct output value for init weight
    output_array[0] = 1.47876192;
    output_array[1] = 1.18172704;
    output_array[2] = 0.81278087;
    output_array[3] = 0.37192339;
    output_array[4] = -0.14084537;
    output_array[5] = 1.5463219;
    output_array[6] = 1.03172099;
    output_array[7] = 0.44520879;
    output_array[8] = -0.2132147;
    output_array[9] = -0.9435495;
    output_array[10] = 1.23600404;
    output_array[11] = 0.50383711;
    output_array[12] = -0.30024111;
    output_array[13] = -1.17623064;
    output_array[14] = -2.12413146;
    test::FillValues<float>(&expected_init_output, output_array);
    // correct output value for trainable weight
    output_array[0] = 5.62282216;
    output_array[1] = 7.15631911;
    output_array[2] = 8.86020461;
    output_array[3] = 10.73447867;
    output_array[4] = 12.77914127;
    output_array[5] = 8.17865041;
    output_array[6] = 10.13811874;
    output_array[7] = 12.26797562;
    output_array[8] = 14.56822105;
    output_array[9] = 17.03885503;
    output_array[10] = 10.73447867;
    output_array[11] = 13.11991837;
    output_array[12] = 15.67574663;
    output_array[13] = 18.40196343;
    output_array[14] = 21.29856878;
    test::FillValues<float>(&expected_trainable_output, output_array);
    // Verify the output tensor.
    //test::ExpectTensorNear<float>(expected_init_output, *GetOutput(0), 1e-4);
    test::ExpectTensorNear<float>(expected_trainable_output, *GetOutput(1), 1e-4);
  }
}

TEST_F(WeightNormGradTest, Norm_Test_Dim0) {
  MakeOpAndSetDevice(Device::CPU, DT_FLOAT);
  // Set up the input tensor.
  AddInputFromArray<float>(TensorShape({3, 5}), {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0});
  AddInputFromArray<float>(TensorShape({3, 5}), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  AddInputFromArray<float>(TensorShape({5, }), {5.0, 4.0, 3.0, 2.0, 1.0});
  AddInput<int64_t>(TensorShape({}), [](int i) { return 0; });
  AddInput<float>(TensorShape({}), [](float i) { return 1e-6; });
  
  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    // Set up the expected output tensor.
    Tensor expected_init_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    Tensor expected_trainable_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    float output_array[15];
    // correct output value for init weight
    output_array[0] = 3.87708297;
    output_array[1] = 2.69075153;
    output_array[2] = 1.76112621;
    output_array[3] = 1.03282878;
    output_array[4] = 0.45816213;
    output_array[5] = 2.01406907;
    output_array[6] = 1.22964452;
    output_array[7] = 0.71720072;
    output_array[8] = 0.37883681;
    output_array[9] = 0.15272071;
    output_array[10] = -2.11477253;
    output_array[11] = -1.75043514;
    output_array[12] = -1.28299239;
    output_array[13] = -0.8135022;
    output_array[14] = -0.38180177;
    test::FillValues<float>(&expected_init_output, output_array);
    // correct output value for trainable weight
    output_array[0] = 15.75203423;
    output_array[1] = 17.9542566;
    output_array[2] = 20.05611961;
    output_array[3] = 22.08299579;
    output_array[4] = 24.05351177;
    output_array[5] = 22.91204978;
    output_array[6] = 25.43519685;
    output_array[7] = 27.77001177;
    output_array[8] = 29.96978;
    output_array[9] = 32.07134903;
    output_array[10] = 30.07206534;
    output_array[11] = 32.91613709;
    output_array[12] = 35.48390393;
    output_array[13] = 37.85656421;
    output_array[14] = 40.08918629;
    test::FillValues<float>(&expected_trainable_output, output_array);
    // Verify the output tensor.
    test::ExpectTensorNear<float>(expected_init_output, *GetOutput(0), 1e-4);
    test::ExpectTensorNear<float>(expected_trainable_output, *GetOutput(1), 1e-4);
  }
}

TEST_F(WeightNormGradTest, Norm_Test_Dim1) {
  MakeOpAndSetDevice(Device::CPU, DT_FLOAT);
  // Set up the input tensor.
  AddInputFromArray<float>(TensorShape({3, 5}), {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0});
  AddInputFromArray<float>(TensorShape({3, 5}), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  AddInputFromArray<float>(TensorShape({5, }), {5.0, 4.0, 3.0, 2.0, 1.0});
  AddInput<int64_t>(TensorShape({}), [](int i) { return 1; });
  AddInput<float>(TensorShape({}), [](float i) { return 1e-6; });
  
  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    // Set up the expected output tensor.
    Tensor expected_init_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    Tensor expected_trainable_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    float output_array[15];
    // correct output value for init weight
    output_array[0] = 6.47231868;
    output_array[1] = 4.41294455;
    output_array[2] = 1.91227597;
    output_array[3] = -1.02968706;
    output_array[4] = -4.41294455;
    output_array[5] = 2.64231304;
    output_array[6] = 1.55969867;
    output_array[7] = 0.33028913;
    output_array[8] = -1.04591558;
    output_array[9] = -2.56891545;
    output_array[10] = 1.8815603;
    output_array[11] = 1.05597772;
    output_array[12] = 0.14719689;
    output_array[13] = -0.84478218;
    output_array[14] = -1.91995949;
    test::FillValues<float>(&expected_init_output, output_array);
    // correct output value for trainable weight
    output_array[0] = 9.25453281;
    output_array[1] = 12.78490324;
    output_array[2] = 16.76344856;
    output_array[3] = 21.19016875;
    output_array[4] = 26.06506382;
    output_array[5] = 13.46113863;
    output_array[6] = 18.11194626;
    output_array[7] = 23.21092877;
    output_array[8] = 28.75808616;
    output_array[9] = 34.75341842;
    output_array[10] = 17.66774445;
    output_array[11] = 23.43898928;
    output_array[12] = 29.65840898;
    output_array[13] = 36.32600357;
    output_array[14] = 43.44177303;
    test::FillValues<float>(&expected_trainable_output, output_array);
    // Verify the output tensor.
    //test::ExpectTensorNear<float>(expected_init_output, *GetOutput(0), 1e-4);
    test::ExpectTensorNear<float>(expected_trainable_output, *GetOutput(1), 1e-4);
  }
}

/*
//----------------------------------------------------------------------------//
//                       Performance benchmarks                               //
//----------------------------------------------------------------------------//
template <typename T>
static Graph* WeightNormGraph(const string& kind, const int64& dim, const TensorShape& shape) {
  auto* graph = new Graph(OpRegistry::Global());

  DataType data_type = DataTypeToEnum<T>::v();

  Tensor input_t(data_type, shape);
  input_t.flat<T>().setRandom();
  Node* input = test::graph::Constant(graph, input_t, "input");

  Tensor trainable_input_t(data_type, shape);
  trainable_input_t.flat<T>().setRandom();
  Node* trainable_input = test::graph::Constant(graph, trainable_input_t, "trainable_input");
  
  Tensor l2norm_axis(DT_INT64, TensorShape({}));
  l2norm_axis.scalar<int64>()() = dim;
  Node* axis = test::graph::Constant(graph, l2norm_axis);

  Tensor l2norm_epsilon(data_type, TensorShape({}));
  l2norm_epsilon.scalar<T>()() = 1e-6;
  Node* eps = test::graph::Constant(graph, l2norm_epsilon);

  const bool isDefault = (kind == "Default");

  // Refer to nn_impl.py, l2_norm for non-complex is calculated using
  // square_sum = math_ops.reduce_sum(math_ops.square(x), axis, keepdims=True)
  // x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon))
  // return math_ops.multiply(x, x_inv_norm, name=name)
  Node* square;
  Node* reduce_sum;
  Node* max;
  Node* rsqrt;
  Node* mul_0;
  Node* mul_1;

  Node* weightnorm;

  if (isDefault) {
    TF_CHECK_OK(NodeBuilder(graph->NewName("square"), "Square")
                    .Input(input)
                    .Attr("T", data_type)
                    .Finalize(graph, &square));

    // To handle dim=2, which should be [0, 1] for reduce_sum axis
    if(dim == 2){
      std::vector<int> all_reduce_dims = {0, 1};
      Tensor all_dim_axis_t(DT_INT64, TensorShape({(int64) all_reduce_dims.size()}));
      auto tensor_data = all_dim_axis_t.flat<int64>().data();
      std::copy(all_reduce_dims.begin(), all_reduce_dims.end(), tensor_data);
      Node* all_dim_axis = test::graph::Constant(graph, all_dim_axis_t);
      TF_CHECK_OK(NodeBuilder(graph->NewName("reduce_sum"), "Sum")
                    .Input(square)
                    .Input(all_dim_axis)
                    .Attr("T", data_type)
                    .Attr("keep_dims", true)
                    .Finalize(graph, &reduce_sum));
    } else {
      TF_CHECK_OK(NodeBuilder(graph->NewName("reduce_sum"), "Sum")
                    .Input(square)
                    .Input(axis)
                    .Attr("T", data_type)
                    .Attr("keep_dims", true)
                    .Finalize(graph, &reduce_sum));
    }

    TF_CHECK_OK(NodeBuilder(graph->NewName("max"), "Maximum")
                    .Input(reduce_sum)
                    .Input(eps)
                    .Attr("T", data_type)
                    .Finalize(graph, &max));

    TF_CHECK_OK(NodeBuilder(graph->NewName("rsqrt"), "Rsqrt")
                    .Input(max)
                    .Attr("T", data_type)
                    .Finalize(graph, &rsqrt));

    TF_CHECK_OK(NodeBuilder(graph->NewName("mul_0"), "Mul")
                    .Input(input)
                    .Input(rsqrt)
                    .Attr("T", data_type)
                    .Finalize(graph, &mul_0));

    TF_CHECK_OK(NodeBuilder(graph->NewName("mul_1"), "Mul")
                    .Input(mul_0)
                    .Input(trainable_input)
                    .Attr("T", data_type)
                    .Finalize(graph, &mul_1));

    return graph;
  } else {
    // weight_norm op.
    TF_CHECK_OK(NodeBuilder(graph->NewName("weight_norm"), "WeightNorm")
                    .Input(input)
                    .Input(trainable_input)
                    .Input(axis)
                    .Input(eps)
                    .Attr("T", data_type)
                    .Finalize(graph, &weightnorm));

    return graph;
  }
}

#define BM_WEIGHTNORM(kind, A, B, type, dim, T)                         \
  static void BM_WEIGHTNORM_##kind##_##A##_##B##_##type##_##dim##_##T(  \
      ::testing::benchmark::State& state) {                             \
    int64 num_computed_elements = (A) * (B);                            \
    int64 flops_per_iter = num_computed_elements;                       \
    test::Benchmark(#type, WeightNormGraph<T>(#kind, dim, {A, B}))      \
        .Run(state);                                                    \
    state.SetItemsProcessed(state.iterations() * flops_per_iter);       \
  }                                                                     \
  BENCHMARK(BM_WEIGHTNORM_##kind##_##A##_##B##_##type##_##dim##_##T)->UseRealTime();

#define BENCHMARK_WEIGHTNORM(A, B, type, T)    \
  BM_WEIGHTNORM(Default, A, B, type, 0, T);    \
  BM_WEIGHTNORM(Fused, A, B, type, 0, T);      \
  BM_WEIGHTNORM(Default, A, B, type, 1, T);    \
  BM_WEIGHTNORM(Fused, A, B, type, 1, T);      \
  BM_WEIGHTNORM(Default, A, B, type, 2, T);    \
  BM_WEIGHTNORM(Fused, A, B, type, 2, T);

#define BENCHMARK_DTYPE(T)                     \
  BENCHMARK_WEIGHTNORM(512, 256, cpu, T);      \
  BENCHMARK_WEIGHTNORM(1024, 512, cpu, T);     \
  BENCHMARK_WEIGHTNORM(1024, 2048, cpu, T);    \
  BENCHMARK_WEIGHTNORM(2048, 1024, cpu, T);    \
  BENCHMARK_WEIGHTNORM(3072, 1024, cpu, T);    \
  BENCHMARK_WEIGHTNORM(4096, 2048, cpu, T);    \
  BENCHMARK_WEIGHTNORM(8192, 4096, cpu, T);

BENCHMARK_DTYPE(float)
BENCHMARK_DTYPE(double)
*/
}  // namespace
} // namespace tensorflow