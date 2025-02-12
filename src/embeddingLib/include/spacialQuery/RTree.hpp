#pragma once

#include <cstdint>
#include <memory>
#include <iterator>
#include <type_traits>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

#include "Graph.hpp"
#include "VecList.hpp"
#include "SpacialIndex.hpp"

// Define a type that is accepted by the boost rtree. Unfortunately, a fixed dimension is enforced
template<size_t D, typename RefType>
struct VecToBoostWrapper {
    RefType value;
};

namespace boost { namespace geometry { namespace traits {

template<size_t D> struct tag<VecToBoostWrapper<D, CVecRef>> { using type = point_tag; };
template<size_t D> struct dimension<VecToBoostWrapper<D, CVecRef>> : boost::mpl::int_<D> {};
template<size_t D> struct coordinate_type<VecToBoostWrapper<D, CVecRef>> { using type = double; };
template<size_t D>  struct coordinate_system<VecToBoostWrapper<D, CVecRef>> { using type = boost::geometry::cs::cartesian; };

template<size_t D> struct tag<VecToBoostWrapper<D, VecRef>> { using type = point_tag; };
template<size_t D> struct dimension<VecToBoostWrapper<D, VecRef>> : boost::mpl::int_<D> {};
template<size_t D> struct coordinate_type<VecToBoostWrapper<D, VecRef>> { using type = double; };
template<size_t D>  struct coordinate_system<VecToBoostWrapper<D, VecRef>> { using type = boost::geometry::cs::cartesian; };

template <size_t D, size_t Index, typename RefType>
struct access<VecToBoostWrapper<D, RefType>, Index> {
    static_assert(Index < D, "Out of range");
    using Point = VecToBoostWrapper<D, RefType>;
    using CoordinateType = double;

    ALWAYS_INLINE static CoordinateType get(Point const& p) {
        ASSERT(D == p.value.dimension());
        return p.value[Index];
    }

    ALWAYS_INLINE static void set(Point& p, const CoordinateType& value) {
        ASSERT(D == p.value.dimension());
        p.value[Index] = value;
    }
};

}}} // namespace boost::geometry::traits



template<typename Point>
class ConstBox {
public:
    ALWAYS_INLINE ConstBox(Point const& min_corner, Point const& max_corner):
        m_min_corner(min_corner), m_max_corner(max_corner) { }

    ALWAYS_INLINE Point const& min_corner() const {
        return m_min_corner;
    }
    ALWAYS_INLINE Point const& max_corner() const {
        return m_max_corner;
    }

private:
    Point m_min_corner;
    Point m_max_corner;
};



namespace boost { namespace geometry
{

// Traits specializations for box above
namespace traits {
template <typename Point>
struct tag<ConstBox<Point> > {
    typedef box_tag type;
};

template <typename Point>
struct point_type<ConstBox<Point> > {
    typedef Point type;
};

template <typename Point, std::size_t Dimension>
struct indexed_access<ConstBox<Point>, min_corner, Dimension> {
    typedef typename geometry::coordinate_type<Point>::type coordinate_type;

    static inline coordinate_type get(ConstBox<Point> const& b) {
        return geometry::get<Dimension>(b.min_corner());
    }
};

template <typename Point, std::size_t Dimension>
struct indexed_access<ConstBox<Point>, max_corner, Dimension>
{
    typedef typename geometry::coordinate_type<Point>::type coordinate_type;

    static inline coordinate_type get(ConstBox<Point> const& b) {
        return geometry::get<Dimension>(b.max_corner());
    }
};

}}}




template<size_t D>
using PointType = VecToBoostWrapper<D, CVecRef>;


namespace impl {
    // This is the black template magic which translates from a runtime dimensions
    // to a compile time dimension

    namespace bgi = boost::geometry::index;

    constexpr size_t MAX_DIMENSION = 16;

    template <size_t D, template<size_t DIM> class PredicateMapper, typename OutIter, typename ValueType, typename... Args>
    struct QueryDispatch;

    class ErasedRTree {
      public:
        template <template<size_t DIM> class PredicateMapper, typename OutIter, typename ValueType, typename... Args>
        ALWAYS_INLINE size_t query(Args... args, OutIter out_it, size_t dimension) const {
            return QueryDispatch<1, PredicateMapper, OutIter, ValueType, Args...>::query(*this, args..., out_it, dimension);
        }

        virtual ~ErasedRTree() = default;
    };

    template<size_t D>
    class RTreeWrapper: public ErasedRTree {
        using RTree = bgi::rtree<std::pair<PointType<D>, NodeId>, bgi::linear<8>>;

        template<typename OutIter, typename ValueType>
        struct MappedOutputIt {
            using iterator_category = typename OutIter::iterator_category;

            MappedOutputIt& operator=(const std::pair<PointType<D>, NodeId>& input) {
                if constexpr (std::is_convertible_v<NodeId, ValueType>) {
                    *iter = input.second;
                } else {
                    std::pair<CVecRef, NodeId> output = std::make_pair(input.first.value, input.second);
                    *iter = output;
                }
                return *this;
            }

            MappedOutputIt& operator*() {
                return *this;
            }

            MappedOutputIt& operator++() {
                ++iter;
                return *this;
            }

            MappedOutputIt operator++(int) {
                return MappedOutputIt<OutIter, ValueType>{iter++};
            }

            bool operator!=(const MappedOutputIt& rhs) {
                return iter != rhs.iter;
            }

            bool operator==(const MappedOutputIt& rhs) {
                return iter == rhs.iter;
            }

            OutIter iter;
        };

      public:
        template<typename Range>
        RTreeWrapper(Range&& r): rtree(std::move(r)) { }

        template <template<size_t DIM> class PredicateMapper, typename OutIter, typename ValueType, typename... Args>
        ALWAYS_INLINE size_t query(Args... args, OutIter out_it) const {
            auto predicates = PredicateMapper<D>::get_predicates(args...);
            MappedOutputIt<OutIter, ValueType> mapped_iterator{out_it};
            return rtree.query(predicates, mapped_iterator);
        }

        RTree rtree;

        ~RTreeWrapper() override = default;
    };

    template <size_t D, template<size_t DIM> class PredicateMapper, typename OutIter, typename ValueType, typename... Args>
    struct QueryDispatch {
        ALWAYS_INLINE static size_t query(const ErasedRTree& rtree, Args... args, OutIter out_it, size_t dimension) {
            if (dimension == D) {
                const RTreeWrapper<D>& concrete = dynamic_cast<const RTreeWrapper<D>&>(rtree);
                return concrete.template query<PredicateMapper, OutIter, ValueType, Args...>(args..., out_it);
            } else {
                return QueryDispatch<D+1, PredicateMapper, OutIter, ValueType, Args...>::query(rtree, args..., out_it, dimension);
            }
        }
    };

    template <template<size_t DIM> class PredicateMapper, typename OutIter, typename ValueType, typename... Args>
    struct QueryDispatch<MAX_DIMENSION, PredicateMapper, OutIter, ValueType, Args...> {
        ALWAYS_INLINE static size_t query(const ErasedRTree& rtree, Args... args, OutIter out_it, size_t dimension) {
            OPTIMIZATION_HINT(MAX_DIMENSION == dimension);
            const RTreeWrapper<MAX_DIMENSION>& concrete = dynamic_cast<const RTreeWrapper<MAX_DIMENSION>&>(rtree);
            return concrete.template query<PredicateMapper, OutIter, ValueType, Args...>(args..., out_it);
        }
    };

    template <size_t D, typename Range>
    ALWAYS_INLINE static std::unique_ptr<ErasedRTree> constructHelper(Range&& r) {
        std::vector<std::pair<PointType<D>, NodeId>> values;
        for (std::pair<CVecRef, NodeId>& input: r) {
            values.push_back(std::make_pair(PointType<D>{input.first}, input.second));
        }
        return std::make_unique<RTreeWrapper<D>>(std::move(values));
    }

    template <size_t D, typename Range>
    struct ConstructionDispatch {
        ALWAYS_INLINE static std::unique_ptr<ErasedRTree> construct(Range&& r, size_t dimension) {
            if (dimension == D) {
                return constructHelper<D>(std::move(r));
            } else {
                return ConstructionDispatch<D+1, Range>::construct(std::move(r), dimension);
            }
        }
    };

    template<typename Range>
    struct ConstructionDispatch<MAX_DIMENSION, Range> {
        ALWAYS_INLINE static std::unique_ptr<ErasedRTree> construct(Range&& r, size_t dimension) {
            OPTIMIZATION_HINT(MAX_DIMENSION == dimension);
            return constructHelper<MAX_DIMENSION>(std::move(r));
        }
    };

    template <typename Range>
    ALWAYS_INLINE std::unique_ptr<ErasedRTree> contructErasedTree(Range&& r, size_t dimension) {
        return ConstructionDispatch<1, Range>::construct(std::move(r), dimension);
    }
}


// define new predicates
namespace predicates {
    namespace bg = boost::geometry;
    namespace bgi = boost::geometry::index;

    template<size_t D>
    struct KNearestPredicates {
        ALWAYS_INLINE static auto get_predicates(CVecRef point, unsigned int number) {
            return bgi::nearest(PointType<D>{point}, number);
        }
    };

    template<size_t D>
    struct InRangePredicates {
        ALWAYS_INLINE static auto get_predicates(CVecRef minCorner, CVecRef maxCorner, CVecRef point, double radius) {
            PointType<D> minC{minCorner};
            PointType<D> maxC{maxCorner};
            PointType<D> p{point};
            ConstBox<PointType<D>> queryBox(minC, maxC);
            return bgi::within(queryBox)
                && bgi::satisfies([=](std::pair<PointType<D>, NodeId> const& v) {
                    return bg::distance(p, v.first) < radius;
            });
        }
    };

    template<size_t D>
    struct InBoxPredicates {
        ALWAYS_INLINE static auto get_predicates(CVecRef minCorner, CVecRef maxCorner) {
            PointType<D> minC{minCorner};
            PointType<D> maxC{maxCorner};
            ConstBox<PointType<D>> queryBox(minC, maxC);
            return bgi::within(queryBox);
        }
    };
}


// the actual usable class
class RTree: public SpatialIndex {
  public:
    template <typename Range>
    RTree(Range&& r, size_t dimension):
        erased_tree(impl::contructErasedTree(std::move(r), dimension)),
        dimension(dimension) {
            ASSERT(dimension >= 1 && dimension <= impl::MAX_DIMENSION);
        }

    size_t query_nearest(CVecRef point, unsigned int number, std::vector<int>& out) const override {
        return query_impl<predicates::KNearestPredicates, std::vector<int>, CVecRef, unsigned int>(
            point, number, out);
    }

    size_t query_sphere(CVecRef point, double radius, std::vector<int>& out) const override {
        VecBuffer<2> buffer(dimension); // TODO(JP): maybe i want to avoid allocation this buffer every time
        
        TmpVec<0> min_corner(buffer);
        TmpVec<1> max_corner(buffer);
        min_corner = point;
        max_corner = point;
        for (int i = 0; i < dimension; i++) {
            min_corner[i] -= radius;
            max_corner[i] += radius;
        }
        return query_impl<predicates::InRangePredicates, std::vector<int>, CVecRef, CVecRef, CVecRef, double>(
            min_corner.erase(), max_corner.erase(), point, radius, out);
    }

    size_t query_box(CVecRef minCorner, CVecRef maxCorner, std::vector<int>& out) const override {
        return query_impl<predicates::InBoxPredicates, std::vector<int>, CVecRef, CVecRef>(
            minCorner, maxCorner, out);
    }

  private:
    template <template<size_t DIM> class PredicateMapper, typename OutputContainer, typename... Args>
    ALWAYS_INLINE size_t query_impl(Args... args, OutputContainer& out) const {
        return erased_tree->query<PredicateMapper, decltype(std::back_inserter(out)), typename OutputContainer::value_type, Args...>(
            args..., std::back_inserter(out), dimension);
    }

    std::unique_ptr<impl::ErasedRTree> erased_tree;
    size_t dimension;
};

