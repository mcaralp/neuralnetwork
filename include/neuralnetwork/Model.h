
#ifndef NEURAL_NETWORK_MODEL_H__
#define NEURAL_NETWORK_MODEL_H__

#include "neuralnetwork/Vector.h"
#include "neuralnetwork/Matrix.h"
#include <cstdint>
#include <tuple>

namespace nn
{
    template<typename... Layers>
    struct ModelCheck
    {
        static constexpr bool value = true;
    };

    template<typename Layer1, typename Layer2, typename... Layers>
    struct ModelCheck<Layer1, Layer2, Layers...>
    {
        static constexpr bool value = std::is_same_v<typename Layer1::Type, typename Layer2::Type> &&
                                    Layer1::Output == Layer2::Input &&
                                    ModelCheck<Layer2, Layers...>::value;
    };

    template<uint32_t I, typename... Layers>
    struct ModelNthLayer
    {
        static_assert(I < sizeof...(Layers), "Invalid index");
    };

    template<typename Layer, typename... Layers>
    struct ModelNthLayer<0, Layer, Layers...>
    {
        using Type = Layer;
    };

    template<uint32_t I, typename Layer, typename... Layers>
    struct ModelNthLayer<I, Layer, Layers...>
    {
        using Type = typename ModelNthLayer<I - 1, Layers...>::Type;
    };

    template<uint32_t I, typename... Layers>
    using ModelNthLayer_t = typename ModelNthLayer<I, Layers...>::Type;

    template<typename... Layers>
    class Model
    {
    public:
        static constexpr uint32_t NbLayers = sizeof...(Layers);

        static_assert(NbLayers > 0, "At least one layer is required");
        static_assert(ModelCheck<Layers...>::value, "Invalid layer configuration");

        static constexpr uint32_t Input = ModelNthLayer_t<0, Layers...>::Input;
        static constexpr uint32_t Output = ModelNthLayer_t<NbLayers - 1, Layers...>::Output;
        using Type = typename ModelNthLayer_t<0, Layers...>::Type;

        template<typename F>
        void fill(F& f)
        {
            fillImpl<0>(f);
        }

        Vector<Type, Output> forward(const Vector<Type, Input>& input)
        {
            return forwardImpl<0>(input);
        }
    private:
        template<uint32_t I, typename F>
        void fillImpl(F& f)
        {
            std::get<I>(m_layers).fill(f);
            if constexpr (I + 1 < NbLayers)
            {
                fillImpl<I + 1>(f);
            }
        }

        template<uint32_t I, uint32_t T>
        Vector<Type, Output> forwardImpl(const Vector<Type, T>& input)
        {
            auto v = std::get<I>(m_layers).forward(input);
            if constexpr (I + 1 < NbLayers)
            {
                return forwardImpl<I + 1>(v);
            }
            else
            {
                return v;
            }
        }

        std::tuple<Layers...> m_layers;
    };
    
}

#endif // NEURAL_NETWORK_MODEL_H__
