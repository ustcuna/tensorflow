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

class WeightNormTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, DataType data_type) {
    // weight_norm op.
    TF_CHECK_OK(NodeDefBuilder("weight_norm", "WeightNorm")
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
TEST_F(WeightNormTest, Norm_Test_All_Dim) {
  MakeOpAndSetDevice(Device::CPU, DT_FLOAT);
  // Set up the input tensor.
  AddInputFromArray<float>(TensorShape({3, 5}), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  AddInputFromArray<float>(TensorShape({1, }), {5.0});
  AddInput<int64_t>(TensorShape({}), [](int i) { return 2; });
  AddInput<float>(TensorShape({}), [](float i) { return 1e-6; });

  // L2 norm should be
  // [[0.02839809 0.05679618 0.08519428 0.11359237 0.14199046]
 //   [0.17038855 0.19878664 0.22718473 0.25558283 0.28398092]
 //   [0.31237901 0.3407771  0.36917519 0.39757328 0.42597138]]
  
  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    // Set up the expected output tensor.
    Tensor expected_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    float output_array[15];
    output_array[0] = 0.14199046;
    output_array[1] = 0.28398092;
    output_array[2] = 0.42597138;
    output_array[3] = 0.56796183;
    output_array[4] = 0.70995229;
    output_array[5] = 0.85194275;
    output_array[6] = 0.99393321;
    output_array[7] = 1.13592367;
    output_array[8] = 1.27791413;
    output_array[9] = 1.41990459;
    output_array[10] = 1.56189504;
    output_array[11] = 1.7038855;
    output_array[12] = 1.84587596;
    output_array[13] = 1.98786642;
    output_array[14] = 2.12985688;
    test::FillValues<float>(&expected_output, output_array);
    // Verify the output tensor.
    test::ExpectTensorNear<float>(expected_output, *GetOutput(0), 1e-4);
  }
}

TEST_F(WeightNormTest, Norm_Test_Dim0) {
  MakeOpAndSetDevice(Device::CPU, DT_FLOAT);
  // Set up the input tensor.
  AddInputFromArray<float>(TensorShape({3, 5}), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  AddInputFromArray<float>(TensorShape({5, }), {5.0, 4.0, 3.0, 2.0, 1.0});
  AddInput<int64_t>(TensorShape({}), [](int i) { return 0; });
  AddInput<float>(TensorShape({}), [](float i) { return 1e-6; });

  // L2 norm should be
  // [[0.07955573 0.1424941  0.1928473  0.2336825  0.26726124]
  //  [0.47733437 0.49872935 0.51425948 0.52578561 0.53452248]
  //  [0.87511301 0.8549646  0.83567165 0.81788873 0.80178373]]
  
  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  

  {
    // Set up the expected output tensor.
    Tensor expected_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    float output_array[15];
    output_array[0] = 0.39777864;
    output_array[1] = 0.5699764;
    output_array[2] = 0.57854191;
    output_array[3] = 0.46736499;
    output_array[4] = 0.26726124;
    output_array[5] = 2.38667185;
    output_array[6] = 1.9949174;
    output_array[7] = 1.54277843;
    output_array[8] = 1.05157123;
    output_array[9] = 0.53452248;
    output_array[10] = 4.37556506;
    output_array[11] = 3.4198584;
    output_array[12] = 2.50701495;
    output_array[13] = 1.63577747;
    output_array[14] = 0.80178373;
    test::FillValues<float>(&expected_output, output_array);
    // Verify the output tensor.
    test::ExpectTensorNear<float>(expected_output, *GetOutput(0), 1e-4);
  }
}

TEST_F(WeightNormTest, Norm_Test_Dim1) {
  MakeOpAndSetDevice(Device::CPU, DT_FLOAT);
  // Set up the input tensor.
  AddInputFromArray<float>(TensorShape({3, 5}), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  AddInputFromArray<float>(TensorShape({3, }), {3.0, 2.0, 1.0});
  AddInput<int64_t>(TensorShape({}), [](int i) { return 1; });
  AddInput<float>(TensorShape({}), [](float i) { return 1e-6; });

  // L2 norm should be
  // [[0.13483997 0.26967994 0.40451992 0.53935989 0.67419986]
  //  [0.33028913 0.38533732 0.44038551 0.49543369 0.55048188]
  //  [0.37619206 0.41039134 0.44459062 0.4787899  0.51298918]]
  
  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    // Set up the expected output tensor.
    Tensor expected_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    float output_array[15];
    output_array[0] = 0.40451992;
    output_array[1] = 0.80903983;
    output_array[2] = 1.21355975;
    output_array[3] = 1.61807967;
    output_array[4] = 2.02259959;
    output_array[5] = 0.66057826;
    output_array[6] = 0.77067464;
    output_array[7] = 0.88077101;
    output_array[8] = 0.99086739;
    output_array[9] = 1.10096377;
    output_array[10] = 0.37619206;
    output_array[11] = 0.41039134;
    output_array[12] = 0.44459062;
    output_array[13] = 0.4787899;
    output_array[14] = 0.51298918;
    test::FillValues<float>(&expected_output, output_array);
    // Verify the output tensor.
    test::ExpectTensorNear<float>(expected_output, *GetOutput(0), 1e-4);
  }
}

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

  Tensor norm_axis(DT_INT64, TensorShape({}));
  norm_axis.scalar<int64>()() = dim;
  Node* axis = test::graph::Constant(graph, norm_axis);

  TensorShape trainable_shape = {1, };
  if(dim == 0){
    trainable_shape = TensorShape({shape.dim_size(1), });
  } else if(dim == 1){
    trainable_shape = TensorShape({shape.dim_size(0), });
  }
  Tensor trainable_input_t(data_type, TensorShape({trainable_shape}));
  trainable_input_t.flat<T>().setRandom();
  Node* trainable_input = test::graph::Constant(graph, trainable_input_t, "trainable_input");

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
  Node* repeat;
  Node* reshape;
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

    if (dim == 2){
      std::vector<int> multiples = {shape.dim_size(0) * shape.dim_size(1)};
      Tensor repeats(DT_INT64, TensorShape({1}));
      auto tensor_data = repeats.flat<int64>().data();
      std::copy(multiples.begin(), multiples.end(), tensor_data);
      Node* repeat_num = test::graph::Constant(graph, repeats);
      TF_CHECK_OK(NodeBuilder(graph->NewName("repeat"), "Tile")
                    .Input(trainable_input)
                    .Input(repeat_num)
                    .Attr("T", data_type)
                    .Finalize(graph, &repeat));

      Tensor reshape_to_shape(DT_INT64, TensorShape({2}));
      std::vector<int> init_shape = {shape.dim_size(0), shape.dim_size(1)};
      auto shape_data = reshape_to_shape.flat<int64>().data();
      std::copy(init_shape.begin(), init_shape.end(), shape_data);
      Node* reshape_to = test::graph::Constant(graph, reshape_to_shape);
      TF_CHECK_OK(NodeBuilder(graph->NewName("reshape"), "Reshape")
                    .Input(repeat)
                    .Input(reshape_to)
                    .Attr("T", data_type)
                    .Finalize(graph, &reshape));

      TF_CHECK_OK(NodeBuilder(graph->NewName("mul_1"), "Mul")
                    .Input(mul_0)
                    .Input(reshape)
                    .Attr("T", data_type)
                    .Finalize(graph, &mul_1));
    } else if (dim == 1){
      std::vector<int> multiples = {shape.dim_size(1)};
      Tensor repeats(DT_INT64, TensorShape({1}));
      auto tensor_data = repeats.flat<int64>().data();
      std::copy(multiples.begin(), multiples.end(), tensor_data);
      Node* repeat_num = test::graph::Constant(graph, repeats);
      TF_CHECK_OK(NodeBuilder(graph->NewName("repeat"), "Tile")
                    .Input(trainable_input)
                    .Input(repeat_num)
                    .Attr("T", data_type)
                    .Finalize(graph, &repeat));

      Tensor reshape_to_shape(DT_INT64, TensorShape({2}));
      std::vector<int> init_shape = {shape.dim_size(0), shape.dim_size(1)};
      auto shape_data = reshape_to_shape.flat<int64>().data();
      std::copy(init_shape.begin(), init_shape.end(), shape_data);
      Node* reshape_to = test::graph::Constant(graph, reshape_to_shape);
      TF_CHECK_OK(NodeBuilder(graph->NewName("reshape"), "Reshape")
                    .Input(repeat)
                    .Input(reshape_to)
                    .Attr("T", data_type)
                    .Finalize(graph, &reshape));
      TF_CHECK_OK(NodeBuilder(graph->NewName("mul_1"), "Mul")
                    .Input(mul_0)
                    .Input(reshape)
                    .Attr("T", data_type)
                    .Finalize(graph, &mul_1));
    } else{
      TF_CHECK_OK(NodeBuilder(graph->NewName("mul_1"), "Mul")
                    .Input(mul_0)
                    .Input(trainable_input)
                    .Attr("T", data_type)
                    .Finalize(graph, &mul_1));
    }

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
  BENCHMARK_WEIGHTNORM(256, 128, cpu, T);      \
  BENCHMARK_WEIGHTNORM(512, 256, cpu, T);      \
  BENCHMARK_WEIGHTNORM(1024, 512, cpu, T);     \
  BENCHMARK_WEIGHTNORM(1024, 2048, cpu, T);    \
  BENCHMARK_WEIGHTNORM(2048, 1024, cpu, T);    \
  BENCHMARK_WEIGHTNORM(3072, 1024, cpu, T);    \
  BENCHMARK_WEIGHTNORM(4096, 2048, cpu, T);    \
  BENCHMARK_WEIGHTNORM(8192, 4096, cpu, T);

BENCHMARK_DTYPE(float)
BENCHMARK_DTYPE(double)

}  // namespace
} // namespace tensorflow