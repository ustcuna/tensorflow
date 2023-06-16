#include "tensorflow/cc/ops/const_op.h"
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
  AddInputFromArray<float>(TensorShape({3, 5}), {15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
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
    output_array[0] = 0.42597138;
    output_array[1] = 0.79514657;
    output_array[2] = 1.10752558;
    output_array[3] = 1.3631084;
    output_array[4] = 1.56189504;
    output_array[5] = 1.7038855;
    output_array[6] = 1.78907978;
    output_array[7] = 1.81747787;
    output_array[8] = 1.78907978;
    output_array[9] = 1.7038855;
    output_array[10] = 1.56189504;
    output_array[11] = 1.3631084;
    output_array[12] = 1.10752558;
    output_array[13] = 0.79514657;
    output_array[14] = 0.42597138;
    test::FillValues<float>(&expected_output, output_array);
    // Verify the output tensor.
    test::ExpectTensorNear<float>(expected_output, *GetOutput(0), 1e-4);
  }
}

TEST_F(WeightNormTest, Norm_Test_Dim0) {
  MakeOpAndSetDevice(Device::CPU, DT_FLOAT);
  // Set up the input tensor.
  AddInputFromArray<float>(TensorShape({3, 5}), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  AddInputFromArray<float>(TensorShape({3, 5}), {15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
  AddInput<int64_t>(TensorShape({}), [](int i) { return 0; });
  AddInput<float>(TensorShape({}), [](float i) { return 1e-6; });

  // L2 norm should be
  // [[0.07955573 0.1424941  0.1928473  0.2336825  0.26726124]
 //   [0.47733437 0.49872935 0.51425948 0.52578561 0.53452248]
 //   [0.87511301 0.8549646  0.83567165 0.81788873 0.80178373]]
  
  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  

  {
    // Set up the expected output tensor.
    Tensor expected_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    float output_array[15];
    output_array[0] = 1.19333593;
    output_array[1] = 1.9949174;
    output_array[2] = 2.50701495;
    output_array[3] = 2.80418994;
    output_array[4] = 2.93987366;
    output_array[5] = 4.77334371;
    output_array[6] = 4.48856415;
    output_array[7] = 4.11407582;
    output_array[8] = 3.6804993;
    output_array[9] = 3.2071349;
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
  AddInputFromArray<float>(TensorShape({3, 5}), {15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
  AddInput<int64_t>(TensorShape({}), [](int i) { return 1; });
  AddInput<float>(TensorShape({}), [](float i) { return 1e-6; });

  // L2 norm should be
  // [[0.13483997 0.26967994 0.40451992 0.53935989 0.67419986]
 //   [0.33028913 0.38533732 0.44038551 0.49543369 0.55048188]
 //   [0.37619206 0.41039134 0.44459062 0.4787899  0.51298918]]
  
  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    // Set up the expected output tensor.
    Tensor expected_output(allocator(), DT_FLOAT, TensorShape({3, 5}));
    float output_array[15];
    output_array[0] = 2.02259959;
    output_array[1] = 3.77551923;
    output_array[2] = 5.25875893;
    output_array[3] = 6.47231868;
    output_array[4] = 7.41619849;
    output_array[5] = 3.3028913;
    output_array[6] = 3.46803586;
    output_array[7] = 3.52308405;
    output_array[8] = 3.46803586;
    output_array[9] = 3.3028913;
    output_array[10] = 1.88096031;
    output_array[11] = 1.64156536;
    output_array[12] = 1.33377186;
    output_array[13] = 0.9575798;
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
static Graph* WeightNormGraph(const int64& dim, const TensorShape& shape) {
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

  Tensor l2norm_epsilon(data_type, TensorShape({}));
  l2norm_epsilon.scalar<T>()() = 1e-6;
  
  Node* weightnorm;
  // weight_norm op.
  TF_CHECK_OK(NodeBuilder(graph->NewName("weight_norm"), "WeightNorm")
                  .Input(input)
                  .Input(trainable_input)
                  .Input(test::graph::Constant(graph, l2norm_axis))
                  .Input(test::graph::Constant(graph, l2norm_epsilon))
                  .Attr("T", data_type)
                  .Finalize(graph, &weightnorm));

  return graph;
}

#define BM_WEIGHTNORM(A, B, type, dim, T)                               \
  static void BM_WEIGHTNORM_##A##_##B##_##type##_##dim##_##T(           \
      ::testing::benchmark::State& state) {                             \
    int64 num_computed_elements = (A) * (B);                            \
    int64 flops_per_iter = num_computed_elements;                       \
    test::Benchmark(#type, WeightNormGraph<T>(dim, {A, B}))             \
        .Run(state);                                                    \
    state.SetItemsProcessed(state.iterations() * flops_per_iter);       \
  }                                                                     \
  BENCHMARK(BM_WEIGHTNORM_##A##_##B##_##type##_##dim##_##T)->UseRealTime();

#define BENCHMARK_WEIGHTNORM(A, B, type, T)    \
  BM_WEIGHTNORM(A, B, type, 2, T);             \
  BM_WEIGHTNORM(A, B, type, 0, T);             \
  BM_WEIGHTNORM(A, B, type, 1, T);

#define BENCHMARK_DTYPE(T)                     \
  BENCHMARK_WEIGHTNORM(32, 64, cpu, T);        \
  BENCHMARK_WEIGHTNORM(64, 128, cpu, T);       \
  BENCHMARK_WEIGHTNORM(128, 256, cpu, T);      \
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