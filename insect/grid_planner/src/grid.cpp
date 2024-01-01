#include "grid.h"
#include <cppitertools/zip.hpp>
#include <cppitertools/range.hpp>
#include <cppitertools/slice.hpp>
#include <cppitertools/enumerate.hpp>
#include <cppitertools/chunked.hpp>
#include <cppitertools/filterfalse.hpp>
#include <cppitertools/count.hpp>

auto operator<<(std::ostream &os, const Grid::Key &k) -> decltype(k.save(os), os)
{
    k.save(os);
    return os;
};
auto operator>>(std::istream &is, Grid::Key &k) -> decltype(k.read(is), is)
{
    k.read(is);
    return is;
};
auto operator<<(std::ostream &os, const Grid::T &t) -> decltype(t.save(os), os)
{
    t.save(os);
    return os;
};
auto operator>>(std::istream &is, Grid::T &t) -> decltype(t.read(is), is)
{
    t.read(is);
    return is;
};

void Grid::initialize(  QRectF dim_,
                        int tile_size,
                        QGraphicsScene *scene_,
                        bool read_from_file,
                        const std::string &file_name,
                        QPointF grid_center,
                        float grid_angle)
{
    static QGraphicsRectItem *bounding_box = nullptr;
    dim = dim_;
    TILE_SIZE = tile_size;
    scene = scene_;
    //qInfo() << __FILE__ << __FUNCTION__ <<  "World dimension: " << dim << TILE_SIZE << "I assume that Y+ axis goes upwards";
    // TODO: CHECK DIMENSIONS BEFORE PROCEED
    //qInfo() << __FUNCTION__ << "Grid coordinates. Center:" << grid_center << "Angle:" << grid_angle;
    for (const auto &[key, value]: fmap)
    {
        scene->removeItem(value.tile);
        delete value.tile;
    }
    if(bounding_box != nullptr) scene->removeItem(bounding_box);
    fmap.clear();

    //    if(read_from_file and not file_name.empty())
    //        readFromFile(file_name);
    auto my_color = QColor("White");
    //my_color.setAlpha(40);
    std::uint32_t id=0;
    Eigen::Matrix2f matrix;
    matrix << std::cos(grid_angle) , -std::sin(grid_angle) , std::sin(grid_angle) , std::cos(grid_angle);
    for(const auto &i: iter::range(dim.left(), dim.right()+TILE_SIZE, static_cast<double>(TILE_SIZE)))
        for(const auto &j: iter::range(dim.top(), dim.bottom()+TILE_SIZE, static_cast<double>(TILE_SIZE)))
        {
            T aux;
            aux.id = id++;
            aux.free = true;
            aux.visited = false;
            aux.cost = 1.0;
            QGraphicsRectItem *tile = scene->addRect(-TILE_SIZE / 2.f, -TILE_SIZE / 2.f, TILE_SIZE, TILE_SIZE,
                                                     QPen(my_color), QBrush(my_color));
            //tile->setZValue(50);
            Eigen::Vector2f res = matrix * Eigen::Vector2f(i, j) + Eigen::Vector2f(grid_center.x(), grid_center.y());
            tile->setPos(res.x(), res.y());
            tile->setRotation(qRadiansToDegrees(grid_angle));
            aux.tile = tile;
            insert(Key(static_cast<long>(i), static_cast<long>(j)), aux);
            keys.emplace_back(i, j);    // list of keys
            //qInfo() << __FUNCTION__ << i << j << aux.id << aux.free << aux.tile->pos();
        }

    // draw bounding box
    bounding_box = scene->addRect(dim, QPen(QColor("Grey"), 40));
    bounding_box->setPos(grid_center);
    bounding_box->setZValue(12);
    bounding_box->setRotation(qRadiansToDegrees(grid_angle));

    qInfo() << __FUNCTION__ <<  "Grid parameters: ";
    qInfo() << "    " << "left:" << dim.left();
    qInfo() << "    " <<"top:" << dim.top();
    qInfo() << "    " << "width:" << dim.width();
    qInfo() << "    " << "height:" << dim.height();
    qInfo() << "    " << "TILE:" << TILE_SIZE;
    qInfo() << "    " << "rows:" << ceil(dim.width() / TILE_SIZE) + 1;
    qInfo() << "    " << "cols:" << ceil(dim.height() / TILE_SIZE) + 1;
    qInfo() << "    " << "elems:" << keys.size() << "(" << (ceil(dim.width() / TILE_SIZE) + 1) * (ceil(dim.height() / TILE_SIZE) + 1) << ")";
}
inline void Grid::insert(const Key &key, const T &value)
{
    fmap.insert(std::make_pair(key, value));
}
inline std::tuple<bool, Grid::T&> Grid::getCell(const Key &k)
{
    if (not dim.contains(k.toQPointF()))
        return std::forward_as_tuple(false, T());
    else
    {
        if (fmap.contains(k))
            return std::forward_as_tuple(true, fmap.at(k));
        else
        {
            //qInfo() << __FUNCTION__ << "Key not found in grid: (" << k.x << k.z << ")";
            // finds the first element with a key not less than k
            auto low_x = std::ranges::lower_bound(keys, k, [](const Key &k, const Key &p)
            { return k.x <= p.x; });
            if (low_x == keys.end() and not keys.empty())
                low_x = std::prev(keys.end());
            std::vector<Key> y_keys;
            std::copy_if(low_x, std::end(keys), std::back_inserter(y_keys), [low_x](const Key &k)
            { return k.x == low_x->x; });
            auto low_y = std::ranges::lower_bound(y_keys, k, [](const Key &k, const Key &p)
            { return k.z < p.z; });     // z is y
            if (low_y != y_keys.end())
            {
                //qWarning() << __FUNCTION__ << " (2) No key found in grid: Requested (" << k.x << k.z << ") but found ("
                //           << low_x->x << low_y->z << ")";
                Key new_key = pointToKey(low_x->x, low_y->z);
                if (fmap.contains(new_key))
                    return std::forward_as_tuple(true, fmap.at(new_key));
                else
                    return std::forward_as_tuple(false, T());
            } else return std::forward_as_tuple(false, T());
        }
    }
}
Grid::Key Grid::pointToKey(long int x, long int z) const
{
    // Key can be invalid for current grid. Check with getCell()
    // bottom is top since Y axis is inverted
    double kx = rint((static_cast<double>(x) - dim.left()) / TILE_SIZE);
    double kz = rint((static_cast<double>(z) - dim.top()) / TILE_SIZE);
    auto k = Key{ static_cast<long>(dim.left() + kx * TILE_SIZE), static_cast<long>(dim.top() + kz * TILE_SIZE)};
    if(not fmap.contains(k))
        qInfo() << __FUNCTION__ << "Key not found in grid: (" << x << z << ") -> (" << k.x << k.z << ")";
    return k;
};
Grid::Key Grid::pointToKey(const QPointF &p) const
{
    // Key can be invalid for current grid. Check with getCell()
    // bottom is top since Y axis is inverted
    double kx = rint((p.x() - dim.left()) / TILE_SIZE);
    double kz = rint((p.y() - dim.top()) / TILE_SIZE);
    auto k = Key{ static_cast<long>(dim.left() + kx * TILE_SIZE), static_cast<long>(dim.top() + kz * TILE_SIZE)};
    if(not fmap.contains(k))
        qInfo() << __FUNCTION__ << "QPoint not found in grid: (" << p.x() << p.y() << ") -> (" << k.x << k.z << ")";
    return k;
};
Grid::Key Grid::pointToKey(const Eigen::Vector2f &p) const
{
    // Key can be invalid for current grid. Check with getCell()
    double kx = ceil((p.x() - dim.left()) / TILE_SIZE);
    double kz = ceil((p.y() - dim.top()) / TILE_SIZE);
    return Key{ static_cast<long>(dim.left() + kx * TILE_SIZE), static_cast<long>(dim.top() + kz * TILE_SIZE)};
};
Eigen::Vector2f Grid::pointToGrid(const Eigen::Vector2f &p) const
{
    return Eigen::Vector2f{ceil((p.x() - dim.left()) / TILE_SIZE), ceil((p.y()) - dim.top()) / TILE_SIZE};
}

///////////////////////////////INPUT / OUTPUT //////////////////////////////////
void Grid::saveToFile(const std::string &fich)
{
    std::ofstream myfile;
    myfile.open(fich);
    for (const auto &[k, v] : fmap)
        myfile << k << v << std::endl;

    myfile.close();
    std::cout << __FUNCTION__ << " " << fmap.size() << " elements written to " << fich << std::endl;
}
std::string Grid::saveToString() const
{
    std::ostringstream stream;
    for (const auto &[k, v] : fmap)
        stream << k << v << v.cost << std::endl;

    std::cout << "Grid::" << __FUNCTION__ << " " << fmap.size() << " elements written to osdtringstream";
    return stream.str();
}
void Grid::readFromString(const std::string &cadena)
{
    fmap.clear();

    std::istringstream stream(cadena);
    std::string line;
    std::uint32_t count = 0;
    while ( std::getline (stream, line) )
    {
        //std::cout << line << std::endl;
        std::stringstream ss(line);
        int x, z;
        bool free, visited;
        float cost;
        std::string node_name;
        ss >> x >> z >> free >> visited >> cost>> node_name;
        fmap.emplace(pointToKey(x, z), T{count++, free, false, cost});
    }
    std::cout << __FUNCTION__ << " " << fmap.size() << " elements read from "  << std::endl;
}
void Grid::readFromFile(const std::string &fich)
{
    std::ifstream myfile(fich);
    std::string line;
    std::uint32_t count = 0;
    if (!myfile)
    {
        std::cout << fich << " No file found" << std::endl;
        std::terminate();
    }
    while ( std::getline (myfile, line) )
    {
        //std::cout << line << std::endl;
        std::stringstream ss(line);
        int x, z;
        bool free, visited;
        std::string node_name;
        ss >> x >> z >> free >> visited >> node_name;
        fmap.emplace(pointToKey(x, z), T{count++, free, false, 1.f});
    }
    std::cout << __FUNCTION__ << " " << fmap.size() << " elements read from " << fich << std::endl;
}

//////////////////////////////// STATUS //////////////////////////////////////////
inline bool Grid::is_free(const Key &k)
{
    const auto &[success, v] = getCell(k);
    if(success)
        return v.free;
    else
        return false;
}
inline bool Grid::is_free(const Eigen::Vector2f &p)
{
    const auto &[success, v] = getCell(pointToKey(static_cast<long int>(p.x()), static_cast<long int>(p.y())));
    if(success)
        return v.free;
    else
        return false;
}
inline bool Grid::is_occupied(const Eigen::Vector2f &p)
{
    const auto &[success, v] = getCell(pointToKey(static_cast<long int>(p.x()), static_cast<long int>(p.y())));
    if(success)
        return not v.free;
    else
        return true;  // non existing cells are returned as occupied
}
void Grid::set_free(const Key &k)
{
    auto &&[success, v] = getCell(k);
    if(success)
    {
        v.free = true;
        if(v.tile != nullptr)
            v.tile->setBrush(QBrush(QColor(params.free_color)));
    }
}
void Grid::set_free(const QPointF &p)
{
    auto x = static_cast<long int>(p.x());
    auto y = static_cast<long int>(p.y());
    set_free(x, y);
}
void Grid::set_free(float xf, float yf)
{
    auto x = static_cast<long int>(xf);
    auto y = static_cast<long int>(yf);
    set_free(x, y);
}
void Grid::set_free(long int x, long int y)
{
    auto &&[success, v] = getCell(pointToKey(x, y));
    if(success)
    {
        v.free = true;
        if (v.tile != nullptr)
            v.tile->setBrush(QBrush(QColor(params.free_color)));
    }
}
//deprecated
void Grid::setOccupied(const Key &k)
{
    auto &&[success, v] = getCell(k);
    if(success)
    {
        v.free = false;
//        if(v.tile != nullptr)
//            v.tile->setBrush(QBrush(QColor(params.occupied_color)));
    }
}
void Grid::setOccupied(long int x, long int y)
{
    auto &&[success, v] = getCell(pointToKey(x, y));
    if(success)
    {
        v.free = false;
//      if(v.tile != nullptr)
//          v.tile->setBrush(QBrush(QColor("red")));
    }
}
void Grid::setOccupied(const QPointF &p)
{
    setOccupied((long int)p.x(), (long int)p.y());
}
void Grid::add_miss_naif(const Eigen::Vector2f &p)
{
    auto &&[success, v] = getCell(pointToKey(static_cast<long int>(p.x()),static_cast<long int>(p.y())));
    if(success)
    {
        v.free = true;
        v.tile->setBrush(QBrush(QColor(params.free_color)));
    }
//    else
//        qWarning() << __FUNCTION__ << "Cell not found" << "[" << p.x() << p.y() << "]";
}
inline void Grid::add_miss(const Eigen::Vector2f &p)
{
    // admissible conditions
    if (not dim.contains(QPointF{p.x(), p.y()}))
        return;

    auto &&[success, v] = getCell(pointToKey(static_cast<long int>(p.x()),static_cast<long int>(p.y())));
    if(success)
    {
        v.misses++;
        if((float)v.hits/(v.hits+v.misses) < params.occupancy_threshold)
        {
            if(not v.free)
                this->flipped++;
            v.free = true;
            v.tile->setBrush(QBrush(QColor(params.free_color)));
        }
        v.misses = std::clamp(v.misses, 0.f, 20.f);
        this->updated++;
    }
////    else
////        qWarning() << __FUNCTION__ << "Cell not found" << "[" << p.x() << p.y() << "]";
}
inline void Grid::add_hit(const Eigen::Vector2f &p)
{
    // admissible conditions
    if (not dim.contains(QPointF{p.x(), p.y()}))
        return;

    auto &&[success, v] = getCell(pointToKey(static_cast<long int>(p.x()),static_cast<long int>(p.y())));
    if(success)
    {
        v.hits++;
        if((float)v.hits/(v.hits+v.misses) >= params.occupancy_threshold)
        {
            if(v.free)
                this->flipped++;
            v.free = false;
            v.tile->setBrush(QBrush(QColor(params.occupied_color)));
        }
        v.hits = std::clamp(v.hits, 0.f, 20.f);
        this->updated++;
    }
}
double Grid::log_odds(double prob)
{
    // Log odds ratio of p(x):
    //              p(x)
    // l(x) = log ----------
    //              1 - p(x)
    return log(prob / (1 - prob));
}
double Grid::retrieve_p(double l)
{
    // Retrieve p(x) from log odds ratio:
    //                   1
    // p(x) = 1 - ---------------
    //             1 + exp(l(x))

    return 1 - 1 / (1 + exp(l));
}
float Grid::percentage_changed() const
{
    return (flipped / updated);
}
void Grid::setVisited(const Key &k, bool visited)
{
    auto &&[success, v] = getCell(k);
    if(success)
    {

        v.visited = visited;
        if(visited)
            v.tile->setBrush(QColor("Orange"));
        else
            v.tile->setBrush(QColor("White"));
    }
}
bool Grid::is_visited(const Key &k)
{
    auto &&[success, v] = getCell(k);
    if(success)
        return v.visited;
    else
        return false;
}
void Grid::setCost(const Key &k,float cost)
{
    auto &&[success, v] = getCell(k);
    if(success)
        v.cost = cost;
}
float Grid::get_cost(const Eigen::Vector2f &p)
{
    auto &&[success, v] = getCell(pointToKey(static_cast<long int>(p.x()),static_cast<long int>(p.y())));
    if(success)
        return v.cost;
    else
        return -1;
}
void Grid::set_all_costs(float value)
{
    for(auto &[key, cell] : fmap)
        cell.cost = value;
}
size_t Grid::count_total() const
{
    return fmap.size();
}
int Grid::count_total_visited() const
{
    int total = 0;
    for(const auto &[k, v] : fmap)
        if(v.visited)
            total ++;
    return total;
}
void Grid::set_all_to_not_visited()
{
    for(auto &[k,v] : fmap)
       setVisited(k, false);
}
void Grid::set_all_to_free()
{
    for(auto &[k,v] : fmap)
        set_free(k);
}
void Grid::markAreaInGridAs(const QPolygonF &poly, bool free)
{
    const qreal step = TILE_SIZE / 4.f;
    QRectF box = poly.boundingRect();
    for (auto &&x : iter::range(box.x() - step / 2.0, box.x() + box.width() + step / 2, step))
        for (auto &&y : iter::range(box.y() - step / 2.0, box.y() + box.height() + step / 2, step))
        {
            if (poly.containsPoint(QPointF(x, y), Qt::OddEvenFill))
            {
                if (free)
                    set_free(pointToKey(static_cast<long>(x), static_cast<long>(y)));
                else
                    setOccupied(pointToKey(static_cast<long>(x), static_cast<long>(y)));
            }
        }
}
void Grid::modifyCostInGrid(const QPolygonF &poly, float cost)
{
    const qreal step = TILE_SIZE / 4.f;
    QRectF box = poly.boundingRect();
    for (auto &&x : iter::range(box.x() - step / 2, box.x() + box.width() + step / 2, step))
        for (auto &&y : iter::range(box.y() - step / 2, box.y() + box.height() + step / 2, step))
            if (poly.containsPoint(QPointF(x, y), Qt::OddEvenFill))
                setCost(pointToKey(static_cast<long>(x), static_cast<long>(y)), cost);
}

////////////////////////////////////// PATH //////////////////////////////////////////////////////////////
std::list<QPointF> Grid::computePath(const QPointF &source_, const QPointF &target_)
{
    // Admission rules
    if (not dim.contains(target_))
    {
        qDebug() << __FUNCTION__ << "Target " << target_.x() << target_.y() << "out of limits " << dim << " Returning empty path";
        return {};
    }
    if (not dim.contains(source_))
    {
        qDebug() << __FUNCTION__ << "Robot out of limits. Returning empty path";
        return {};
    }
    Key target_key = pointToKey(target_);
    const auto &[succ_trg, target_cell] = getCell(target_key);
    if(not succ_trg)
    {
        qWarning() << "Could not find target position in Grid. Returning empty path";
        return {};
    }
    Key source_key = pointToKey(source_);
    const auto &[succ_src, source_cell] = getCell(source_key);
    if(not succ_src)
    {
        qWarning() << "Could not find source position in Grid. Returning empty path";
        return {};
    }
    if (source_key == target_key)
    {
        qDebug() << __FUNCTION__ << "Robot already at target. Returning empty path";
        return {};
    }

    // vector de distancias inicializado a UINT_MAX
    std::vector<uint32_t> min_distance(fmap.size(), std::numeric_limits<uint32_t>::max());
    // initialize source position to 0
    min_distance[source_cell.id] = 0;
    // vector de pares<std::uint32_t, Key> initialized to (-1, Key())
    std::vector<std::pair<std::uint32_t, Key>> previous(fmap.size(), std::make_pair(-1, Key()));
    // lambda to compare two vertices: a < b if a.id<b.id or
    auto comp = [this](std::pair<std::uint32_t, Key> x, std::pair<std::uint32_t, Key> y){ return x.first <= y.first; };

    // OPEN List
    std::vector<std::vector<Eigen::Vector2f>> paths_list;
    std::set<std::pair<std::uint32_t, Key>, decltype(comp)> active_vertices(comp);
    active_vertices.insert({0, source_key});
    while (not active_vertices.empty())
    {
        Key where = active_vertices.begin()->second;
        if (where == target_key)  // target found
        {
            auto p = orderPath(previous, source_key, target_key);
            p = decimate_path(p);  // reduce size of path to half
            return p;
        }
        active_vertices.erase(active_vertices.begin());
        for (auto ed : neighboors_8(where))
        {
            //qInfo() << __FUNCTION__ << min_distance[ed.second.id] << ">" << min_distance[fmap.at(where).id] << "+" << ed.second.cost;
            const auto &[succ, where_cell] = getCell(where);
            if (min_distance[ed.second.id] > min_distance[where_cell.id] + static_cast<uint32_t>(ed.second.cost))
            {
                active_vertices.erase({min_distance[ed.second.id], ed.first});
                min_distance[ed.second.id] = min_distance[where_cell.id] + static_cast<uint32_t>(ed.second.cost);
                min_distance[ed.second.id] = min_distance[where_cell.id] + static_cast<uint32_t>(ed.second.cost);
                previous[ed.second.id] = std::make_pair(where_cell.id, where);
                active_vertices.insert({min_distance[ed.second.id], ed.first}); // Djikstra
                //active_vertices.insert( { min_distance[ed.second.id] + heuristicL1(ed.first, target_key), ed.first } ); //A*
            }
        }
    }
    qInfo() << __FUNCTION__ << "Path from (" << source_key.x << "," << source_key.z << ") to (" <<  target_.x() << "," << target_.y() << ") not  found. Returning empty path";
    return {};
};
std::vector<std::vector<Eigen::Vector2f>> Grid::compute_k_paths(const QPointF &source_, const QPointF &target_, int num_paths)
{
    // lambda to convert a list of QPointF to a vector of Eigen::Vector2f
    auto to_eigen = [](const std::list<QPointF> &path)
            {
                std::vector<Eigen::Vector2f> path_v(path.size());
                for(const auto &[i, p] : iter::enumerate(path))
                    path_v[i] = Eigen::Vector2f{static_cast<float>(p.x()), static_cast<float>(p.y())};
                return path_v;
            };

    // Yen's algorithm
    // get an initial shortest path
    auto i_path = computePath(source_, target_);
    if(i_path.empty()) return {};

    auto initial_path = to_eigen(i_path);
    // proceed by setting as occupied the cells of the path and searching for an alternative one
    int k= 0;
    std::vector<std::vector<Eigen::Vector2f>> paths_list;
    paths_list.push_back(initial_path);
    auto current_step= initial_path.cbegin();   // source
    Key deleted_key = pointToKey(source_);
    while(k < num_paths)
    {
        // restore previously cell set to occupied
        set_free(deleted_key);
        // get next key from path and mark it as occupied in the grid
        if(current_step = std::next(current_step); current_step != initial_path.cend())
        {
            auto key = pointToKey(*current_step);
            setOccupied(key);
            auto path = computePath(source_, target_);
            if(not path.empty())
            {
                auto path_eigen = to_eigen(path);
                auto mdist = max_distance(initial_path, path_eigen);
                if(mdist > 100)
                {
                    paths_list.emplace_back(path_eigen);
                    k++;
                    //qInfo() << __FUNCTION__ << "Max distance: " << mdist << " Path length: " << path.size();
                }
            }
        }
        else
            break;
    }
    return paths_list;
}
std::vector<Eigen::Vector2f> Grid::compute_path(const QPointF &source_, const QPointF &target_)
{
    auto lpath = computePath(source_, target_);
    std::vector<Eigen::Vector2f> path(lpath.size());
    for(const auto &[i, p] : iter::enumerate(lpath))
        path[i] = Eigen::Vector2f{static_cast<float>(p.x()), static_cast<float>(p.y())};
    return  path;
}
std::vector<std::pair<Grid::Key, Grid::T>> Grid::neighboors(const Grid::Key &k, const std::vector<int> &xincs,const std::vector<int> &zincs,
                                                            bool all)
{
    std::vector<std::pair<Key, T>> neigh;
    // list of increments to access the neighboors of a given position
    for (auto &&[itx, itz]: iter::zip(xincs, zincs))
    {
        Key lk{k.x + itx, k.z + itz};
        auto &&[success, p] = getCell(lk);
        if (not success) continue;

        // check that incs are not both zero but have the same abs value, i.e. a diagonal
//        if (itx != 0 and itz != 0 and (fabs(itx) == fabs(itz)) and p.cost == 1)
//            p.cost = 1.43;                                // if neighboor in diagonal, cost is sqrt(2)

        if (all)
            neigh.emplace_back(lk, p);
        else // if all cells covered by the robot are free
        {
            //bool all_free = true;
            if (p.free)
                neigh.emplace_back(lk, p);
//            {
//                if(ceil(400.0/TILE_SIZE)<= 3) // robot occupies three cells, Check 8-neigh
//                {
//                    auto neigh = neighboors_8(lk, true);
//                    if( auto res = std::ranges::find_if_not(neigh, [](auto a){ return a.second.free;}); res != neigh.end())
//                        all_free = false;
////                    for (auto &&[fitx, fitz]: iter::zip(xincs, zincs))
////                    {
////                        Key flk{lk.x + fitx, lk.z + fitz};
////                        const auto &[fsuccess, fp] = getCell(flk);
////                        if (not fsuccess or not fp.free)
////                        {
////                            all_free = false;
////                            break;
////                        }
////                    }
//                }
//                else
//                {
//                    auto neigh = neighboors_16(lk, true);
//                    if( auto res = std::ranges::find_if_not(neigh, [](auto a){ return a.second.free;}); res != neigh.end())
//                        all_free = false;
//                }
//                if (all_free)
//                    neigh.emplace_back(std::make_pair(lk, p));
//            }
        }
    }
    return neigh;
}
std::vector<std::pair<Grid::Key, Grid::T>> Grid::neighboors_8(const Grid::Key &k, bool all)
{
    const int &I = TILE_SIZE;
    static const std::vector<int> xincs = {I, I, I, 0, -I, -I, -I, 0};
    static const std::vector<int> zincs = {I, 0, -I, -I, -I, 0, I, I};
    return this->neighboors(k, xincs, zincs, all);
}
std::vector<std::pair<Grid::Key, Grid::T>> Grid::neighboors_16(const Grid::Key &k, bool all)
{
    const int &I = TILE_SIZE;
    static const std::vector<int> xincs = {0,   I,   2*I,  2*I, 2*I, 2*I, 2*I, I, 0, -I, -2*I, -2*I,-2*I,-2*I,-2*I, -I};
    static const std::vector<int> zincs = {2*I, 2*I, 2*I,  I,   0 , -I , -2*I, -2*I,-2*I,-2*I,-2*I, -I, 0,I, 2*I, 2*I};
    return this->neighboors(k, xincs, zincs, all);
}
std::list<QPointF> Grid::orderPath(const std::vector<std::pair<std::uint32_t, Key>> &previous, const Key &source, const Key &target)
{
    // recovers the path from the previous vector
    std::list<QPointF> res;
    Key k = target;
    std::uint32_t u = fmap.at(k).id;
    while (previous[u].first != (std::uint32_t)-1)
    {
        res.emplace_front(static_cast<double>(k.x), static_cast<long>(k.z));
        u = previous[u].first;
        k = previous[u].second;
    }
    //qDebug() << __FILE__ << __FUNCTION__ << "Path length:" << res.size();  //exit point
    return res;
};
std::list<QPointF> Grid::decimate_path(const std::list<QPointF> &path)
{
    std::list<QPointF> res;
    for(auto &&p : iter::chunked(path,2))
        res.push_back(p[0]);
    return res;
}
inline double Grid::heuristicL2(const Key &a, const Key &b) const
{
    return std::hypot(a.x - b.x, a.z - b.z);
}
inline double Grid::heuristicL1(const Key &a, const Key &b) const
{
    return std::abs(a.x - b.x) + std::abs(a.z - b.z);
}

/////////////////////////////// COSTS /////////////////////////////////////////////////////////
void Grid::update_costs(float robot_semi_width, bool wide)
{
    static QBrush free_brush(QColor(params.free_color));
    static QBrush occ_brush(QColor(params.occupied_color));
    static QBrush orange_brush(QColor("Orange"));
    static QBrush yellow_brush(QColor("Yellow"));
    static QBrush gray_brush(QColor("LightGray"));
    static QBrush green_brush(QColor("LightGreen"));
    static std::vector<std::tuple<float, float, QBrush, std::function<std::vector<std::pair<Grid::Key, Grid::T>>(Grid*, Grid::Key, bool)>>> wall_ranges
                ={{100, 75, orange_brush, &Grid::neighboors_8},
                  {75, 50, yellow_brush, &Grid::neighboors_8},
                  {50, 25, gray_brush, &Grid::neighboors_8},
                  {25, 10,  green_brush, &Grid::neighboors_16}};

    // we assume the grid has been cleared before. All free cells have cost 1

    // if not free, set cost to 100. These are cells detected by the  Lidar.
    for (auto &&[k, v]: iter::filterfalse([](auto &v) { return std::get<1>(v).free; }, fmap))
    {
        v.cost = 100;
        v.tile->setBrush(occ_brush);
    }

    if(wide)    // if wide is true, we set the cost of the cells detected by the Lidar to 100 and their neighboors to values defined in wall_ranges
    {
        for(auto &[upper, lower, brush, neigh] : wall_ranges)
            // get all cells with cost == upper
            for (auto &&[k, v]: iter::filter([upper, lower](auto &v) { return std::get<1>(v).cost == upper; }, fmap))
                // get all neighboors of these cells whose cost is lower than upper and are free
                for (auto neighs = neigh(this, k, false); auto &&[kk, vv]: neighs | iter::filter([upper](auto &ve)
                                                                                           { return std::get<1>(ve).cost < upper and std::get<1>(ve).free; }))
                {
                    const auto &[ok, cell] = getCell(kk);
                    cell.cost = lower;
                    cell.free = true;
                    cell.tile->setBrush(brush);
                }
    }
}
void Grid::update_map( const std::vector<Eigen::Vector3f> &points, const Eigen::Vector2f &robot_in_grid, float max_laser_range)
{
    for(const auto &point : points)
    {
        float length = (point.head(2)-robot_in_grid).norm();
        int num_steps = ceil(static_cast<double>(length/(static_cast<float>(TILE_SIZE))));
        Eigen::Vector2f p;
        for(const auto &&step : iter::range(0.0, 1.0-(1.0/num_steps), 1.0/num_steps))
        {
            p = robot_in_grid * (1-step) + point.head(2)*step;
            add_miss(p);
        }
        if(length <= max_laser_range)
            add_hit(point.head(2));

        if((p-point.head(2)).norm() < static_cast<float>(TILE_SIZE))  // in case last miss overlaps tip
            add_hit(point.head(2));
    }
}

////////////////////////////// DRAW /////////////////////////////////////////////////////////
void Grid::draw(bool clear) // if clear is true, only removes the previous points
{
    //clear previous points
    static std::vector<QGraphicsRectItem *> scene_grid_points;
    for (QGraphicsRectItem* item : scene_grid_points)
    {
        scene->removeItem((QGraphicsItem *) item);
        delete item;
    }
    scene_grid_points.clear();

    if(not clear)
    {
        //create new representation
        std::string color;
        for (const auto &[key, value]: fmap)
        {
            if (value.free)
            {
                if (value.cost == 2.0) //affordance spaces
                    color = "#FFFF00";
                else if (value.cost == 3.0) //lowvisited spaces
                    color = "#FFBF00";
                else if (value.cost == 4.0) //mediumvisited spaces
                    color = "#FF8000";
                else if (value.cost == 5.0) //highVisited spaces
                    color = "#FF4000";
                else if (value.cost == 8.0) //zona social
                    color = "#BF00FF";
                else if (value.cost == 10.0) //zona personal
                    color = "#00BFFF";
                else if (value.cost == 50.0) //Affordance maximum
                    color = "#FF0000";
                else
                    color = "White";
            } else // occupied
                color = "Red";

            QColor my_color = QColor(QString::fromStdString(color));
            my_color.setAlpha(40);
            QGraphicsRectItem *aux = scene->addRect(-TILE_SIZE / 2, -TILE_SIZE / 2, TILE_SIZE, TILE_SIZE,
                                                    QPen(my_color), QBrush(my_color));
            aux->setZValue(1);
            aux->setPos(key.x, key.z);
            scene_grid_points.push_back(aux);
        }
    }
}
void Grid::clear()
{
    for (auto &[key, value]: fmap)
    {
        value.tile->setBrush(QBrush(QColor(params.free_color)));
        value.free = true;
        value.cost = 1;
    }
}
void Grid::reset()
{
    for (auto &[key, value]: fmap)
    {
        scene->removeItem(value.tile);
        delete value.tile;
    }
    keys.clear();
    fmap.clear();
}

////////////////////////////// QUERIES /////////////////////////////////////////////////////////
float Grid::max_distance(const std::vector<Eigen::Vector2f> &pathA, const std::vector<Eigen::Vector2f> &pathB)
{
    // Approximates Frechet distance
    std::vector<float> dists;
    for(auto &&i: iter::range(std::min(pathA.size(), pathB.size())))
        dists.emplace_back((pathA[i] - pathB[i]).norm());
    return std::ranges::max(dists);
}
float Grid::frechet_distance(const std::vector<Eigen::Vector2f> &A, const std::vector<Eigen::Vector2f> &B)
{
    // Frechet distance between to paths
    int n = A.size(), m = B.size();
    Eigen::MatrixXf dp(n, m);
    dp(0, 0) = (A[0] - B[0]).norm();

    // Fill first row and column
    for (int i = 1; i < n; ++i)
    {
        dp(i, 0) = std::max(dp(i - 1, 0), (A[i] - B[0]).norm());
    }
    for (int j = 1; j < m; ++j)
    {
        dp(0, j) = std::max(dp(0, j - 1), (A[0] - B[j]).norm());
    }

    // Fill rest of the table
    for (int i = 1; i < n; ++i)
    {
        for (int j = 1; j < m; ++j)
        {
            float minPrevious = std::min({dp(i - 1, j), dp(i, j - 1), dp(i - 1, j - 1)});
            dp(i, j) = std::max(minPrevious, (A[i] - B[j]).norm());
        }
    }
    return dp(n - 1, m - 1);
}
std::optional<QPointF> Grid::closestMatching_spiralMove(const QPointF &p, const std::function<bool(std::pair<Grid::Key, Grid::T>)> &pred)
{
    if(not dim.adjusted(-500, -500, 500, 500).contains(p))  // TODO: remove this hack
    {
        qInfo() << __FUNCTION__ << "Point " << p.x() << p.y() << "out of limits " << dim;
        return {};
    }

    const auto &[ok, cell] = getCell(pointToKey(p));

    // if not ok, return empty
    //    if(not ok)
    //        return {};

    // if free, return point
    if(ok and cell.free)
        return p;

    int move_unit = TILE_SIZE;
    int vi = move_unit;
    int vj = 0;
    int tam_segmento = 1;
    int i = static_cast<int>(p.x()), j = static_cast<int>(p.y());
    int recorrido = 0;

    QPointF ret_point;
    while(true)
    {
        i += vi; j += vj; ++recorrido;
        ret_point.setX(i); ret_point.setY(j);
        Key key = pointToKey(ret_point);
        const auto &[success, v] = getCell(key);
        if(success and pred(std::make_pair(key, v)))
            return ret_point;
        if (recorrido == tam_segmento)
        {
            recorrido = 0;
            int aux = vi; vi = -vj; vj = aux;
            if (vj == 0)
                ++tam_segmento;
        }
    }
}
std::optional<QPointF> Grid::closest_obstacle(const QPointF &p)
{
    return this->closestMatching_spiralMove(p, [](auto cell){ return not cell.second.free; });
}
std::optional<QPointF> Grid::closest_free(const QPointF &p)
{
    return this->closestMatching_spiralMove(p, [](auto cell){ return cell.second.free; });
}
std::optional<QPointF> Grid::closest_free_4x4(const QPointF &p)
{
    return this->closestMatching_spiralMove(p, [this, p](const auto &cell){
        if (not cell.second.free)
            return false;
        Key key = pointToKey(QPointF(cell.first.x, cell.first.z));
        std::vector<std::pair<Grid::Key, Grid::T>> L1 = neighboors_16(key, false);
        return (L1.size() == 16);
    });
}
bool Grid::is_path_blocked(const std::vector<Eigen::Vector2f> &path) // grid coordinates
{
    return std::ranges::any_of(path, [this](const auto &p){ return is_occupied(p);});
}
std::tuple<bool, QVector2D> Grid::vectorToClosestObstacle(QPointF center)
{
    auto k = pointToKey(center);
    QVector2D closestVector;
    bool obstacleFound = false;

    auto neigh = neighboors_8(k, true);
    float dist = std::numeric_limits<float>::max();
    for (auto n : neigh)
    {
        if (not n.second.free)
        {
            QVector2D vec = QVector2D(QPointF(k.x, k.z)) - QVector2D(QPointF(n.first.x,n.first.z)) ;
            if (vec.length() < dist)
            {
                dist = vec.length();
                closestVector = vec;
            }
            qDebug() << __FUNCTION__ << "Obstacle found";
            obstacleFound = true;
        }
    }

    if (!obstacleFound)
    {
        auto DistNeigh = neighboors_16(k, true);
        for (auto n : DistNeigh)
        {
            if (not n.second.free)
            {
                QVector2D vec = QVector2D(QPointF(k.x, k.z)) - QVector2D(QPointF(n.first.x, n.first.z)) ;
                if (vec.length() < dist)
                {
                    dist = vec.length();
                    closestVector = vec;
                }
                obstacleFound = true;
            }
        }
    }
    return std::make_tuple(obstacleFound,closestVector);
}
bool Grid::line_of_sigth_to_target(const Eigen::Vector2f &source, const Eigen::Vector2f &target)
{
    // Admission rules
    if (not dim.contains(QPointF{target.x(), target.y()}))
    {
        qInfo() << "[GRID]" << __FUNCTION__ << "Target " << target.x() << target.y() << "out of limits " << dim;
        return false;
    }
    if (not dim.contains(QPointF{source.x(), source.y()}))
    {
        qInfo() << "[GRID]" << __FUNCTION__ << "Source " << target.x() << target.y() << "out of limits " << dim;
        return false;
    }

    // check if there is a straight line from source to target that is free
    float num_steps = (target - source).norm() / static_cast<float>(TILE_SIZE);
    Eigen::Vector2f step((target - source) / num_steps);

    // compute how many parallel lines we need to cover the robot's width
    int num_lines_to_side = ceil(250.0 / TILE_SIZE);
    bool success = true;
    for (auto &&i: iter::range(-num_lines_to_side, num_lines_to_side + 1, 1))
    {
        Eigen::Vector2f src = source - Eigen::Vector2f{-TILE_SIZE * i, 0.f};
        success = success and std::ranges::all_of(iter::range(0.f, num_steps, 1.f), [this, src, step](auto &&i)
                        { return is_free(src + step * i); });
    }
    return success;
}
