#pragma once

#include "DmlDFT.h"

class GpuSTFTOperator : public WRL::Base<IMLOperatorKernel>
{
private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;

    int64_t m_axis;
    bool m_isOnesided;
    bool m_isInverse;

public:
    GpuSTFTOperator(IMLOperatorKernelCreationContext* context)
    {
        ComPtr<IUnknown> executionObject;
        context->GetExecutionInterface(executionObject.GetAddressOf());

        ComPtr<IMLOperatorKernelCreationContextNodeWrapperPrivate> contextPrivateNodeWrapper;
        context->QueryInterface(IID_PPV_ARGS(&contextPrivateNodeWrapper));

        ComPtr<IUnknown> dmlProviderI;
        ORT_THROW_IF_FAILED(contextPrivateNodeWrapper->GetExecutionProvider(dmlProviderI.GetAddressOf()));

        ComPtr<ID3D12GraphicsCommandList> commandList;
        ORT_THROW_IF_FAILED(executionObject.As(&commandList));

        ComPtr<Dml::IExecutionProvider> dmlProvider;
        dmlProviderI.As(&dmlProvider);

        ORT_THROW_IF_FAILED(commandList->GetDevice(IID_ID3D12Device, &m_device));

        int64_t isOnesidedInt;
        ORT_THROW_IF_FAILED(context->GetAttribute("onesided", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&isOnesidedInt)));
        m_isOnesided = static_cast<bool>(isOnesidedInt);

        // TODO: create DML element-wise & DFT ops
    }

    // Computes the outputs of the kernel.  This may be called multiple times
    // simultaneously within the same instance of the class.  Implementations
    // of this method must be thread-safe.
    STDMETHOD(Compute)(IMLOperatorKernelContext* context)
    {
        // TODO: element-wise dispatch
        // DFT::Compute
        return S_OK;
    }
};

struct STFTShapeInferrer : public WRL::Base<IMLOperatorShapeInferrer>
{
    STDMETHOD(InferOutputShapes)(IMLOperatorShapeInferenceContext* context) noexcept
    {
        try
        {
            ComPtr<IMLOperatorShapeInferenceContextPrivate> contextPrivate;
            ORT_THROW_IF_FAILED(context->QueryInterface(IID_PPV_ARGS(&contextPrivate)));

            int64_t isOnesidedInt;
            ORT_THROW_IF_FAILED(context->GetAttribute("onesided", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&isOnesidedInt)));

            bool isOnesided = static_cast<bool>(isOnesidedInt);

            // input 0: signal
            uint32_t batchSize = 0;
            uint32_t signalLength = 0;
            {
                uint32_t rank;
                ORT_THROW_IF_FAILED(context->GetInputTensorDimensionCount(0, &rank));
                if (rank == 0) { return E_FAIL; }

                std::vector<uint32_t> inputDims(rank);
                ORT_THROW_IF_FAILED(context->GetInputTensorShape(0, rank, inputDims.data()));
                batchSize = inputDims[0];
                signalLength = inputDims[1];
            }

            // input 1: frame_step (required; constant)
            uint32_t frameStep = 0;
            {
                ComPtr<IMLOperatorTensor> frameStepTensorInterface;
                ORT_THROW_IF_FAILED(contextPrivate->GetConstantInputTensor(1, &frameStepTensorInterface));
                MLOperatorTensor frameStepTensor(frameStepTensorInterface.Get());
                frameStep = onnxruntime::narrow<uint32_t>(OperatorHelper::ReadScalarTensorCastToInt64(frameStepTensor));
            }

            // input 2: window (optional)
            uint32_t windowLength = 0;
            if (context->IsInputValid(2))
            {
                uint32_t rank;
                ORT_THROW_IF_FAILED(context->GetInputTensorDimensionCount(2, &rank));
                if (rank != 1) { return E_FAIL; }

                ORT_THROW_IF_FAILED(context->GetInputTensorShape(2, rank, &windowLength));
            }

            // input 3: frame_length (optional; constant)
            uint32_t frameLength = 0;
            if (context->IsInputValid(3))
            {
                ComPtr<IMLOperatorTensor> frameLengthTensorInterface;
                ORT_THROW_IF_FAILED(contextPrivate->GetConstantInputTensor(3, &frameLengthTensorInterface));
                MLOperatorTensor frameLengthTensor(frameLengthTensorInterface.Get());
                frameLength = onnxruntime::narrow<uint32_t>(OperatorHelper::ReadScalarTensorCastToInt64(frameLengthTensor));
            }

            // Frame/window lengths represent the same quantity, so use whichever is set (non-zero).
            windowLength = std::max(windowLength, frameLength);
            frameLength = std::max(windowLength, frameLength);

            const uint32_t frameCount = (signalLength - windowLength) / frameStep + 1;
            const uint32_t uniqueBins = isOnesided ? frameLength / 2 + 1 : frameLength;

            std::array<uint32_t, 4> outputDims = { batchSize, frameCount, uniqueBins, 2 };

            ORT_THROW_IF_FAILED(context->SetOutputTensorShape(0, onnxruntime::narrow<uint32_t>(outputDims.size()), outputDims.data()));
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }
};

class GpuSTFTOperatorFactory : public WRL::Base<IMLOperatorKernelFactory>
{
public:
    STDMETHOD(CreateKernel)(
        IMLOperatorKernelCreationContext* context,
        IMLOperatorKernel** kernel)
    {
        try
        {
            auto dftOperator = wil::MakeOrThrow<GpuSTFTOperator>(context);
            dftOperator.CopyTo(kernel);
            return S_OK;
        }
        catch (...)
        {
            return E_FAIL;
        }
    }

    static void RegisterSTFTKernel(IMLOperatorRegistry* registry)
    {
        MLOperatorKernelDescription kernelDescription = {};
        kernelDescription.domain = "";
        kernelDescription.name = "STFT";
        kernelDescription.minimumOperatorSetVersion = 17;
        kernelDescription.executionType = MLOperatorExecutionType::D3D12;

        // T1: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
        MLOperatorEdgeTypeConstrant t1Constraint;
        t1Constraint.typeLabel = "T1";
        std::vector<MLOperatorEdgeDescription> t1AllowedEdges
        {
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float },
        };
        t1Constraint.allowedTypes = t1AllowedEdges.data();
        t1Constraint.allowedTypeCount = static_cast<uint32_t>(t1AllowedEdges.size());

        // T2 : tensor(int32), tensor(int64)
        MLOperatorEdgeTypeConstrant t2Constraint;
        t2Constraint.typeLabel = "T2";
        std::vector<MLOperatorEdgeDescription> t2AllowedEdges
        {
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Int64 },
        };
        t2Constraint.allowedTypes = t2AllowedEdges.data();
        t2Constraint.allowedTypeCount = static_cast<uint32_t>(t2AllowedEdges.size());

        std::vector<MLOperatorEdgeTypeConstrant> typeConstraints{ t1Constraint, t2Constraint };
        kernelDescription.typeConstraints = typeConstraints.data();
        kernelDescription.typeConstraintCount = static_cast<uint32_t>(typeConstraints.size());

        MLOperatorAttributeNameValue onesidedAttributeValue;
        onesidedAttributeValue.name = "onesided";
        onesidedAttributeValue.type = MLOperatorAttributeType::Int;
        onesidedAttributeValue.valueCount = 1;
        static const int64_t onesided[] = { 1 };
        onesidedAttributeValue.ints = onesided;

        std::vector<MLOperatorAttributeNameValue> attributeDefaultValues{ onesidedAttributeValue };

        kernelDescription.defaultAttributes = attributeDefaultValues.data();
        kernelDescription.defaultAttributeCount = static_cast<uint32_t>(attributeDefaultValues.size());
        kernelDescription.options = MLOperatorKernelOptions::None;
        kernelDescription.executionOptions = 0;

        auto shareInferrer = wil::MakeOrThrow<STFTShapeInferrer>();
        auto factory = wil::MakeOrThrow<GpuSTFTOperatorFactory>();

        std::array<uint32_t, 3> requiredConstantCpuInputs = { /*frame_step*/1, /*frame_length*/3 };

        ComPtr<IMLOperatorRegistryPrivate> registryPrivate;
        ORT_THROW_IF_FAILED(registry->QueryInterface(IID_PPV_ARGS(&registryPrivate)));

        ORT_THROW_IF_FAILED(registryPrivate->RegisterOperatorKernel(
            &kernelDescription,
            factory.Get(),
            shareInferrer.Get(),
            nullptr,
            false, // isInternalOperator
            false, // alias
            false, // supportsGraph
            nullptr,
            requiredConstantCpuInputs.data(),
            static_cast<uint32_t>(requiredConstantCpuInputs.size())
        ));
    }
};
