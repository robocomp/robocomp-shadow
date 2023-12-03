//
// Created by pbustos on 22/12/22.
//

#include "graph.h"
#include <cppitertools/range.hpp>
#include <cppitertools/enumerate.hpp>

void rc::Graph::init(AbstractGraphicViewer *viewer_)
{
    viewer = viewer_;
    connect(viewer, SIGNAL(new_mouse_coordinates(QPointF)), this, SLOT(left_button_event(QPointF)));
}

int rc::Graph::add_node()
{
    int id = id_counter;
    nodes.insert(std::make_pair(id, Node(id)));
    id_counter++;
    qInfo() << __FUNCTION__ << "Created node " << id;
    return id;
}

int rc::Graph::add_node(int node_dest)
{
    if(nodes.contains(node_dest))
    {
        int id = add_node();
        edges.insert(std::make_pair(std::make_pair(id, node_dest), Edge(id, node_dest)));
        return id;
    }
    else
    {
        qWarning() << __FUNCTION__ << "Warning: destination node not existing or valid";
        return -1;
    }
}

void rc::Graph::add_edge(int n1, int n2)
{
    edges.insert(std::make_pair(std::make_pair(n1, n2), Edge(n1, n2)));
    qInfo() << __FUNCTION__ << "Created edge between nodes " << n1 << "and " << n2;
}

void rc::Graph::draw()
{
    if(viewer == nullptr) return;
    static std::vector<QGraphicsItem *> items;
    for (auto i: items)
    {
        viewer->scene.removeItem(i);
        delete i;
    }
    items.clear();

    const int max_cols = 6;
    //int rows_step = viewer->height() / 4;
    qInfo() << __FUNCTION__ << viewer->scene.sceneRect().width() << viewer->scene.sceneRect().height();
    int rows_step = viewer->scene.sceneRect().height() / 2;
    int cols_step = viewer->scene.sceneRect().width() / (2 * max_cols);
    int rows = rows_step, cols = cols_step;
    //nodes
    for (auto &&[i, node]: nodes | iter::enumerate)
    {
        qInfo() << __FUNCTION__ << rows << cols;
//        if ((i + 1) % (max_cols + 1) == 0)
//        {
//            rows = rows + 2 * rows_step;
//            cols = cols_step;
//        } else
        auto &[k, v] = node;
        v.draw_pos.setX(cols);
        v.draw_pos.setY(rows);
        auto item = viewer->scene.addEllipse(-100, -100, 200, 200, QPen(QColor("brown"), 5));
        item->setPos(v.draw_pos);
        items.push_back(item);
        cols = cols + 2*cols_step;

        auto item_t = viewer->scene.addSimpleText(QString::number(v.objects.size()), QFont("times", 40));
        item_t->setPos(v.draw_pos - QPointF(10, 30));
        items.push_back(item_t);
    }
    //edges
    for (auto &&[i, edge]: edges | iter::enumerate)
    {
        auto &[k, v] = edge;
        auto &[n1, n2] = k;
        auto p1 = nodes.at(n1).draw_pos;
        auto p2 = nodes.at(n2).draw_pos;
        auto minx = std::min(p1.x(), p2.x());
        auto maxx = std::max(p1.x(), p2.x());
        auto item = viewer->scene.addLine(minx+100, p1.y(), maxx-100, p2.y(), QPen(QColor("brown"), 10));
        items.push_back(item);
    }
}

void rc::Graph::add_tags(int id, const std::vector<rc::PreObject> &objects)
{
    if(nodes.contains(id))
    {
        auto n = nodes.at(id);
        for(const auto &o: objects)
            nodes.at(id).objects.insert(o.label);
    }
}

///////////////////////////////////////////////////////////
void rc::Graph::left_button_event(QPointF p)
{
    qInfo() << __FUNCTION__ << p;
    // select closest node to p coor
    // show a modal window with the list of labels and other node attributes
}
