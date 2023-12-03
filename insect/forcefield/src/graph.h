//
// Created by pbustos on 22/12/22.
//

#ifndef FORCEFIELD_GRAPH_H
#define FORCEFIELD_GRAPH_H

#include <abstract_graphic_viewer/abstract_graphic_viewer.h>
#include <Eigen/Dense>
#include <QtCore>
#include "preobject.h"
#include <QObject>


namespace rc
{
    class Graph : QObject
    {
        Q_OBJECT
        public:
            Graph() = default;
            void init(AbstractGraphicViewer *viewer_= nullptr);
            struct Node
            {
                Node(int id_) : id(id_){};
                int id;
                std::set<std::string> objects;
                QPointF draw_pos{0.f, 0.f};
            };
            struct Edge
            {
                Edge(int n1_, int n2_) : n1(n1_), n2(n2_){};
                int n1, n2;
            };
            int add_node();
            int add_node(int node_dest);
            void add_edge(int n1, int n2);
            void add_tags(int id, const std::vector<rc::PreObject> &objects);
            void draw();

        public slots:
            void left_button_event(QPointF);

        private:
            int id_counter = 0;
            std::map<int, Node> nodes;
            std::map<std::pair<int, int>,  Edge> edges;
            AbstractGraphicViewer *viewer;

    };
}; //rc

#endif //FORCEFIELD_GRAPH_H
