#pragma once

#include <array>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <type_traits>
#include <vector>

#include "Macros.hpp"
#include "Rand.hpp"
#include "Toolkit.hpp"

namespace impl {

constexpr int DIMENSION = 2;

// some forward declarations
template <typename Inner, int SLOT>
class VecRefImpl;

template <typename Inner, int SLOT>
class CVecRefImpl;

template <typename L, typename R>
class AddExpr;

template <typename L, typename R>
class SubExpr;

template <typename E>
class MultExpr;

template <typename Inner>
class VecBase {
    using Self = VecBase<Inner>;

   public:
    using ExprType = Self;

    ALWAYS_INLINE unsigned int dimension() const {
        return coord.dimension();
    }

    ALWAYS_INLINE double operator[](int i) const {
        return coord[i];
    }

    ALWAYS_INLINE double norm() const {
        return std::sqrt(sqNorm());
    }

    ALWAYS_INLINE double infNorm() const {
        return coord.infNorm();
    }

    ALWAYS_INLINE double sqNorm() const {
        return coord.sqNorm();
    }

    std::string toString() const {
        std::string result = "[";
        for (int i = 0; i < dimension(); i++) {
            result += std::to_string(coord[i]) + " ";
        }
        result += "]";
        return result;
    }

   protected:
    using ChunkType = typename Inner::ChunkType;

    template <typename I, int S>
    friend class VecRefImpl;
    template <typename I, int S>
    friend class CVecRefImpl;
    template <typename L, typename R>
    friend class AddExpr;
    template <typename L, typename R>
    friend AddExpr<typename L::ExprType, typename R::ExprType> operator+(const L&, const R&);
    template <typename L, typename R>
    friend class SubExpr;
    template <typename L, typename R>
    friend SubExpr<typename L::ExprType, typename R::ExprType> operator-(const L&, const R&);
    template <typename E>
    friend class MultExpr;
    template <typename E>
    friend MultExpr<typename E::ExprType> operator*(double, const E&);

    VecBase() : coord() {
        // fail hard in case of invalid access
        if constexpr (ASSERTIONS_ACTIVE) {
            coord.poison();
        }
    }

    VecBase(const Inner& coord) : coord(coord) {}

    VecBase(const Self&) = default;
    VecBase(Self&&) = default;

    Self& operator=(const Self&) = default;
    Self& operator=(Self&&) = default;

    ALWAYS_INLINE ChunkType chunkAt(unsigned int i) const {
        return coord.chunkAt(i);
    }

    Inner coord;
};

// operator definitions
template <typename L, typename R>
struct AddExpr {
    using ChunkType = typename L::ChunkType;
    using ExprType = AddExpr<L, R>;
    static_assert(std::is_same_v<ChunkType, typename R::ChunkType>);

    L left;
    R right;

    ALWAYS_INLINE ChunkType chunkAt(unsigned int i) const {
        ChunkType result = left.chunkAt(i);
        result += right.chunkAt(i);
        return result;
    }
};

template <typename L, typename R>
ALWAYS_INLINE AddExpr<typename L::ExprType, typename R::ExprType> operator+(const L& left, const R& right) {
    return AddExpr<typename L::ExprType, typename R::ExprType>{left, right};
}

template <typename L, typename R>
struct SubExpr {
    using ChunkType = typename L::ChunkType;
    using ExprType = SubExpr<L, R>;
    static_assert(std::is_same_v<ChunkType, typename R::ChunkType>);

    L left;
    R right;

    ALWAYS_INLINE ChunkType chunkAt(unsigned int i) const {
        ChunkType result = left.chunkAt(i);
        result -= right.chunkAt(i);
        return result;
    }
};

template <typename L, typename R>
ALWAYS_INLINE SubExpr<typename L::ExprType, typename R::ExprType> operator-(const L& left, const R& right) {
    return SubExpr<typename L::ExprType, typename R::ExprType>{left, right};
}

template <typename E>
struct MultExpr {
    using ChunkType = typename E::ChunkType;
    using ExprType = MultExpr<E>;

    double scalar;
    E vec;

    ALWAYS_INLINE ChunkType chunkAt(unsigned int i) const {
        ChunkType result = vec.chunkAt(i);
        result *= scalar;
        return result;
    }
};

template <typename E>
ALWAYS_INLINE MultExpr<typename E::ExprType> operator*(double scalar, const E& vec) {
    return MultExpr<typename E::ExprType>{scalar, vec};
}

template <typename E>
ALWAYS_INLINE MultExpr<typename E::ExprType> operator*(const E& vec, double scalar) {
    return scalar * vec;
}

// buffer for temporary vector
template <typename Inner, unsigned int N_SLOTS>
struct VecBuffer {
    using MemoryType = typename Inner::MemoryType;

    VecBuffer(unsigned int dimension) : dim(dimension), buffer(N_SLOTS * Inner::numEntriesForDimension(dim)) {
#ifdef EMBEDDING_USE_ASSERTIONS
        ref_counts.resize(N_SLOTS, 0);
#endif
    }

    unsigned int dimension() const {
        return dim;
    }

    template <typename T, unsigned int SLOT>
    T construct() {
        static_assert(SLOT < N_SLOTS);
        return T(&buffer[SLOT * Inner::numEntriesForDimension(dim)], dim);
    }

    unsigned int dim;
    std::vector<MemoryType> buffer;

#ifdef EMBEDDING_USE_ASSERTIONS
    std::vector<int> ref_counts;
#endif
};

template <typename Inner, unsigned int N_SLOTS>
struct DummyBuffer {
    DummyBuffer(unsigned int dimension) {
        ASSERT(dimension == this->dimension(),
               "Dimension " << dimension << " not supported by the current Vec type.");
#ifdef EMBEDDING_USE_ASSERTIONS
        ref_counts.resize(N_SLOTS, 0);
#endif
    }

    unsigned int dimension() const {
        return Inner().dimension();
    }

    template <typename T, unsigned int>
    T construct() {
        return T();
    }

#ifdef EMBEDDING_USE_ASSERTIONS
    std::vector<int> ref_counts;
#endif
};

// the user-facing vec types
template <typename Inner, int SLOT>
class VecRefImpl {
    static_assert(SLOT >= -1);
    using Vec = VecBase<Inner>;
    using Self = VecRefImpl<Inner, SLOT>;
    // Note: a temporary vector has the same semantics as a VecRef but might contain the
    // data directly. Therefore, we make a compile-time case distinction here.
    using ContainedType = std::conditional_t<SLOT == -1, typename Inner::RefType, typename Inner::TmpValueType>;

    template <unsigned int N_SLOTS>
    using Buffer = typename Inner::BufferType<N_SLOTS>;

   public:
    using MemoryType = typename Inner::MemoryType;
    using ExprType = Vec;

    VecRefImpl(MemoryType* mem, unsigned int dimension) : coord(mem, dimension) {}

    template <unsigned int N_SLOTS>
    VecRefImpl(Buffer<N_SLOTS>& buffer) : coord(buffer.template construct<ContainedType, SLOT>()) {
        static_assert(SLOT >= 0, "TmpVec with negative slot not allowed!");
        static_assert(SLOT < N_SLOTS, "Invalid slot!");
        if constexpr (ASSERTIONS_ACTIVE) {
            initRefCount(buffer);
            // poison the contained values
            setAll(std::nan(""));
        }
    }

    template <unsigned int N_SLOTS>
    VecRefImpl(Buffer<N_SLOTS>& buffer, double default_value) : coord(buffer.template construct<ContainedType, SLOT>()) {
        static_assert(SLOT >= 0, "TmpVec with negative slot not allowed!");
        static_assert(SLOT < N_SLOTS, "Invalid slot!");
        setAll(default_value);
        if constexpr (ASSERTIONS_ACTIVE) {
            initRefCount(buffer);
        }
    }

    // copying a VecRef is error-prone and thus forbidden
    VecRefImpl(const Self&) = delete;

    VecRefImpl(Self&& other) : VecRefImpl(std::move(other)) {}

    ~VecRefImpl() {
        if constexpr (ASSERTIONS_ACTIVE) {
            decreaseRefCount();
        }
    }

    ALWAYS_INLINE Self& operator=(const Self& other) {
        *this = static_cast<Vec>(other);
        return *this;
    }

    template <typename Expr>
    ALWAYS_INLINE Self& operator=(const Expr& expr) {
        auto e = static_cast<typename Expr::ExprType>(expr);
        for (size_t i = 0; i < coord.get().numChunks(); ++i) {
            coord.get().chunkAt(i) = e.chunkAt(i);
        }
        return *this;
    }

    ALWAYS_INLINE unsigned int dimension() const {
        return static_cast<Vec>(*this).dimension();
    }

    ALWAYS_INLINE void setAll(double value) {
        coord.get().setAll(value);
    }

    ALWAYS_INLINE double operator[](int i) const {
        return static_cast<Vec>(*this)[i];
    }

    ALWAYS_INLINE double& operator[](int i) {
        return coord.get()[i];
    }

    ALWAYS_INLINE double norm() const {
        return static_cast<Vec>(*this).norm();
    }

    ALWAYS_INLINE double infNorm() const {
        return static_cast<Vec>(*this).infNorm();
    }

    ALWAYS_INLINE double sqNorm() const {
        return static_cast<Vec>(*this).sqNorm();
    }

    template <typename Expr>
    ALWAYS_INLINE Self& operator+=(const Expr& expr) {
        auto e = static_cast<typename Expr::ExprType>(expr);
        for (size_t i = 0; i < coord.get().numChunks(); ++i) {
            coord.get().chunkAt(i) += e.chunkAt(i);
        }
        return *this;
    }

    template <typename Expr>
    ALWAYS_INLINE Self& operator-=(const Expr& expr) {
        auto e = static_cast<typename Expr::ExprType>(expr);
        for (size_t i = 0; i < coord.get().numChunks(); ++i) {
            coord.get().chunkAt(i) -= e.chunkAt(i);
        }
        return *this;
    }

    ALWAYS_INLINE Self& operator*=(const double scalar) {
        for (size_t i = 0; i < coord.get().numChunks(); ++i) {
            coord.get().chunkAt(i) *= scalar;
        }
        return *this;
    }

    ALWAYS_INLINE Self& operator/=(const double scalar) {
        for (size_t i = 0; i < coord.get().numChunks(); ++i) {
            coord.get().chunkAt(i) /= scalar;
        }
        return *this;
    }

    ALWAYS_INLINE void cWiseMax(const double scalar) {
        for (int i = 0; i < this->dimension(); i++) {
            coord.get()[i] = std::max(scalar, coord.get()[i]);
        }
    }

    ALWAYS_INLINE void cWiseMin(const double scalar) {
        for (int i = 0; i < this->dimension(); i++) {
            coord.get()[i] = std::min(scalar, coord.get()[i]);
        }
    }

    ALWAYS_INLINE void normed() {
        double norm = this->norm();
        if (norm > 0) {
            for (int i = 0; i < this->dimension(); i++) {
            coord.get()[i] /= norm;
            }
        }
    }

    /**
     * Returns the derivative of the vector according to the inf norm.
     * 
     * This vector only consits of zeros and +1/-1.
     * The only nonnegative value is at the position of the maximum value.
     */
    ALWAYS_INLINE void infNormed() {
        int maxIndex = this->maxIndex();
        double sign = coord.get()[maxIndex] > 0 ? 1 : -1;
        for (int i = 0; i < this->dimension(); i++) {
            coord.get()[i] = 0;
        }
        coord.get()[maxIndex] = sign;
    }

    /**
     * Returns the dimension that has the largest absolute value
    */
    ALWAYS_INLINE int maxIndex() const {
        int maxIndex = 0;
        double max = 0;
        for(int i = 0; i < this->dimension(); i++) {
            if(std::abs(coord.get()[i]) > max) {
                max = std::abs(coord.get()[i]);
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    ALWAYS_INLINE void setToRandomUnitVector() {
        for (int i = 0; i < dimension(); i++) {
            coord.get()[i] = Rand::gaussDistribution(0.0, 1.0);
        }
        double length = norm();
        double radius = Toolkit::myPow(Rand::randomDouble(0.0, 1.0), 1.0 / (double)dimension());
        for (int i = 0; i < dimension(); i++) {
            coord.get()[i] *= radius / length;
        }
    }

    std::string toString() const {
        return static_cast<Vec>(*this).toString();
    }

    ALWAYS_INLINE operator Vec() const {
        return Vec(coord.get());
    }

    ALWAYS_INLINE VecRefImpl<Inner, -1> erase() {
        return VecRefImpl<Inner, -1>(std::move(*this));
    }

    ALWAYS_INLINE static constexpr size_t numEntriesForDimension(unsigned int dimension) {
        return Inner::numEntriesForDimension(dimension);
    }

   private:
    template <int S>
    VecRefImpl(VecRefImpl<Inner, S>&& other) : coord(other.coord) {
        if constexpr (ASSERTIONS_ACTIVE) {
            moveRefCount(other);
        }
        other.moveOut();
    }

    template <unsigned int N_SLOTS>
    void initRefCount(Buffer<N_SLOTS>& buffer) {
#ifdef EMBEDDING_USE_ASSERTIONS
        ref_count = &buffer.ref_counts[SLOT];
        ASSERT(*ref_count == 0, "Double use of slot " << SLOT);
        (*ref_count)++;
#endif
    }

    template <int S>
    void moveRefCount(const VecRefImpl<Inner, S>& other) {
#ifdef EMBEDDING_USE_ASSERTIONS
        ref_count = other.ref_count;
#endif
    }

    void decreaseRefCount() {
#ifdef EMBEDDING_USE_ASSERTIONS
        if (ref_count != nullptr) {
            (*ref_count)--;
            ASSERT(*ref_count == 0);
        }
#endif
    }

    ALWAYS_INLINE void moveOut() {
#ifdef EMBEDDING_USE_ASSERTIONS
        ref_count = nullptr;
        // fail hard in case of invalid access
        coord.poison();
#endif
    }

    template <typename I, int S>
    friend class VecRefImpl;
    template <typename I, int S>
    friend class CVecRefImpl;

    ContainedType coord;
#ifdef EMBEDDING_USE_ASSERTIONS
    int* ref_count = nullptr;
#endif
};

template <typename Inner, int SLOT>
class CVecRefImpl : public VecBase<Inner> {
    using Base = VecBase<Inner>;
    using Self = CVecRefImpl<Inner, SLOT>;
    using Base::coord;

   public:
    using MemoryType = typename Inner::MemoryType;

    CVecRefImpl() : Base() {}

    CVecRefImpl(const MemoryType* mem, unsigned int dimension) : Base(Inner(mem, dimension)) {}

    CVecRefImpl(const Self& other) : Base() {
        *this = other;
    }

    CVecRefImpl(Self&& other) : CVecRefImpl(static_cast<const Self&>(other)) {}

    CVecRefImpl(VecRefImpl<Inner, SLOT>&& mut_ref) : Base() {
        *this = std::move(mut_ref);
    }

    ALWAYS_INLINE Self& operator=(const Self& other) {
        if constexpr (ASSERTIONS_ACTIVE) {
            moveRefCount(other);
            increaseRefCount();
        }
        coord = other.coord;
        return *this;
    }

    ALWAYS_INLINE Self& operator=(Self&& other) {
        return (*this = static_cast<const Self&>(other));
    }

    ALWAYS_INLINE Self& operator=(VecRefImpl<Inner, SLOT>&& mut_ref) {
        if constexpr (ASSERTIONS_ACTIVE) {
            moveRefCount(mut_ref);
        }
        coord = mut_ref.coord.get();
        mut_ref.moveOut();
        return *this;
    }

    ALWAYS_INLINE CVecRefImpl<Inner, -1> erase() const {
        CVecRefImpl<Inner, -1> erased(static_cast<Base>(*this));
        if constexpr (ASSERTIONS_ACTIVE) {
            erased.moveRefCount(*this);
            erased.increaseRefCount();
        }
        return erased;
    }

    ~CVecRefImpl() {
        if constexpr (ASSERTIONS_ACTIVE) {
            decreaseRefCount();
        }
    }

   private:
    template <typename I, int S>
    friend class CVecRefImpl;

    CVecRefImpl(const Base& base) : Base(base) {}

    template <typename T>
    void moveRefCount(const T& other) {
#ifdef EMBEDDING_USE_ASSERTIONS
        ref_count = other.ref_count;
#endif
    }

    void increaseRefCount() {
#ifdef EMBEDDING_USE_ASSERTIONS
        if (ref_count != nullptr) {
            (*ref_count)++;
        }
#endif
    }

    void decreaseRefCount() {
#ifdef EMBEDDING_USE_ASSERTIONS
        if (ref_count != nullptr) {
            (*ref_count)--;
        }
#endif
    }

#ifdef EMBEDDING_USE_ASSERTIONS
    int* ref_count = nullptr;
#endif
};

template <typename T, typename Construct>
struct ValueImpl {
    using MemoryType = ValueImpl<T, Construct>;
    using ChunkType = T;
    using RefType = struct WrappedPtr {
        WrappedPtr(MemoryType* ptr, unsigned int dimension) : wrapped(ptr) {
            ASSERT(ptr->dimension() == dimension);
            unused(dimension);
        }

        WrappedPtr(const WrappedPtr& other) = default;

        ALWAYS_INLINE ValueImpl<T, Construct>& get() {
            return *wrapped;
        }

        ALWAYS_INLINE const ValueImpl<T, Construct>& get() const {
            return *wrapped;
        }

        void poison() {
            wrapped = nullptr;
        }

        ValueImpl<T, Construct>* wrapped;
    };
    using TmpValueType = ValueImpl<T, Construct>;

    template <unsigned int N_SLOTS>
    using BufferType = DummyBuffer<ValueImpl<T, Construct>, N_SLOTS>;

    ValueImpl(const ValueImpl&) = default;
    ValueImpl& operator=(const ValueImpl&) = default;

    ValueImpl() : data(Construct::construct()) {}

    ValueImpl(const MemoryType* ptr, unsigned int dimension) : ValueImpl(*ptr) {
        ASSERT(ptr->dimension() == dimension);
        unused(dimension);
    }

    ALWAYS_INLINE unsigned int dimension() const {
        return data.size();
    }

    ALWAYS_INLINE double& operator[](int i) {
        ASSERT(i < dimension());
        return data[i];
    }

    ALWAYS_INLINE double operator[](int i) const {
        ASSERT(i < dimension());
        return data[i];
    }

    ALWAYS_INLINE void setAll(double value) {
        data.setConstant(value);
    }

    ALWAYS_INLINE double sqNorm() const {
        return data.squaredNorm();
    }

    ALWAYS_INLINE double infNorm() const {
        // NOTE: maybe this is inefficient?
        double max = 0;
        for (int i = 0; i < dimension(); i++) {
            max = std::max(std::abs(data[i]), max);
        }
        return max;
    }

    ALWAYS_INLINE ValueImpl& get() {
        return *this;
    }

    ALWAYS_INLINE const ValueImpl& get() const {
        return *this;
    }

    ALWAYS_INLINE ChunkType chunkAt(unsigned int i) const {
        ASSERT(i == 0);
        unused(i);
        return data;
    }

    ALWAYS_INLINE ChunkType& chunkAt(unsigned int i) {
        ASSERT(i == 0);
        unused(i);
        return data;
    }

    ALWAYS_INLINE size_t numChunks() const {
        return 1;
    }

    ALWAYS_INLINE static constexpr size_t numEntriesForDimension(unsigned int dimension) {
        ASSERT(dimension == Construct::construct().size());
        return 1;
    }

    void poison() {
        for (size_t i = 0; i < dimension(); i++) {
            data[i] = std::nan("");
        }
    }

   private:
    T data;
};

// own implementation of basic operations in case we don't want to use Eigen
template <unsigned int D>
struct ArrayBaseType {
    ArrayBaseType(const ArrayBaseType&) = default;
    ArrayBaseType& operator=(const ArrayBaseType&) = default;

    ArrayBaseType(const std::array<double, D>& data) : data(data) {}

    ALWAYS_INLINE unsigned int size() const {
        return D;
    }

    ALWAYS_INLINE double& operator[](int i) {
        ASSERT(i < D);
        return data[i];
    }

    ALWAYS_INLINE double operator[](int i) const {
        ASSERT(i < D);
        return data[i];
    }

    ALWAYS_INLINE void setConstant(double value) {
        for (int i = 0; i < D; i++) {
            data[i] = value;
        }
    }

    ALWAYS_INLINE ArrayBaseType& operator+=(const ArrayBaseType& other) {
        for (int i = 0; i < D; i++) {
            data[i] += other.data[i];
        }
        return *this;
    }

    ALWAYS_INLINE ArrayBaseType& operator-=(const ArrayBaseType& other) {
        for (int i = 0; i < D; i++) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    ALWAYS_INLINE ArrayBaseType& operator*=(const double scalar) {
        for (int i = 0; i < D; i++) {
            data[i] *= scalar;
        }
        return *this;
    }

    ALWAYS_INLINE ArrayBaseType& operator/=(const double scalar) {
        for (int i = 0; i < D; i++) {
            data[i] /= scalar;
        }
        return *this;
    }

    ALWAYS_INLINE double squaredNorm() const {
        double sum = 0;
        for (int i = 0; i < D; i++) {
            sum += data[i] * data[i];
        }
        return sum;
    }

    ALWAYS_INLINE double infNorm() const {
        double max = 0;
        for (int i = 0; i < D; i++) {
            max = std::max(std::abs(data[i]), max);
        }
        return max;
    }

   private:
    std::array<double, D> data;
};

using Eigen::Vector;
using Eigen::VectorXd;

template <unsigned int D>
struct ConstructVector {
    static constexpr Vector<double, D> construct() {
        return Vector<double, D>();
    }
};

template <unsigned int D>
struct ConstructArray {
    static constexpr std::array<double, D> construct() {
        return std::array<double, D>();
    }
};

struct IndirectionImpl {
    using MemoryType = double;
    using ChunkType = double;
    using RefType = IndirectionImpl;
    using TmpValueType = IndirectionImpl;

    template <unsigned int N_SLOTS>
    using BufferType = VecBuffer<IndirectionImpl, N_SLOTS>;

    IndirectionImpl(const IndirectionImpl&) = default;
    IndirectionImpl& operator=(const IndirectionImpl&) = default;

    IndirectionImpl() : ptr(nullptr), dim(0) {}

    IndirectionImpl(unsigned int dimension) : ptr(nullptr), dim(dimension) {}

    IndirectionImpl(MemoryType* ptr, unsigned int dimension) : ptr(ptr), dim(dimension) {}

    IndirectionImpl(const MemoryType* ptr, unsigned int dimension) : IndirectionImpl(const_cast<MemoryType*>(ptr), dimension) {}

    ALWAYS_INLINE unsigned int dimension() const {
        return dim;
    }

    ALWAYS_INLINE double& operator[](int i) {
        ASSERT(i < dim);
        return ptr[i];
    }

    ALWAYS_INLINE double operator[](int i) const {
        ASSERT(i < dim);
        return ptr[i];
    }

    ALWAYS_INLINE void setAll(double value) {
        for (int i = 0; i < dim; i++) {
            ptr[i] = value;
        }
    }

    ALWAYS_INLINE double sqNorm() const {
        double sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += ptr[i] * ptr[i];
        }
        return sum;
    }

    ALWAYS_INLINE double infNorm() const {
        double max = 0;
        for (int i = 0; i < dim; i++) {
            max = std::max(std::abs(ptr[i]), max);
        }
        return max;
    }


    ALWAYS_INLINE IndirectionImpl& get() {
        return *this;
    }

    ALWAYS_INLINE const IndirectionImpl& get() const {
        return *this;
    }

    ALWAYS_INLINE ChunkType chunkAt(unsigned int i) const {
        ASSERT(i < dim);
        return ptr[i];
    }

    ALWAYS_INLINE ChunkType& chunkAt(unsigned int i) {
        ASSERT(i < dim);
        return ptr[i];
    }

    ALWAYS_INLINE size_t numChunks() const {
        return dim;
    }

    ALWAYS_INLINE static constexpr size_t numEntriesForDimension(unsigned int dimension) {
        return dimension;
    }

    void poison() {
        ptr = nullptr;
    }

   private:
    double* ptr;
    unsigned int dim;
};

// explicitely instantiate the different variants to trigger compiler
// errors if one does not work
template class VecBase<ValueImpl<ArrayBaseType<DIMENSION>, ConstructArray<DIMENSION>>>;
template class VecBase<ValueImpl<Eigen::Vector<double, DIMENSION>, ConstructVector<DIMENSION>>>;
template class VecBase<ValueImpl<Eigen::VectorXd, ConstructVector<DIMENSION>>>;
template class VecBase<IndirectionImpl>;

}  // end of namespace impl

// change this declaration to exchange the vector type

// using InnerType = impl::ValueImpl<impl::ArrayBaseType<impl::DIMENSION>, impl::ConstructArray<impl::DIMENSION>>;
// using InnerType = impl::ValueImpl<Eigen::Vector<double, impl::DIMENSION>, impl::ConstructVector<impl::DIMENSION>>;
// using InnerType = impl::ValueImpl<Eigen::VectorXd, impl::ConstructVector<impl::DIMENSION>>;
using InnerType = impl::IndirectionImpl;

using CVecRef = impl::CVecRefImpl<InnerType, -1>;
using VecRef = impl::VecRefImpl<InnerType, -1>;

template <unsigned int SLOT>
using TmpVec = impl::VecRefImpl<InnerType, SLOT>;
template <unsigned int SLOT>
using TmpCVec = impl::CVecRefImpl<InnerType, SLOT>;
template <unsigned int SLOT>
using VecBuffer = typename InnerType::BufferType<SLOT>;
