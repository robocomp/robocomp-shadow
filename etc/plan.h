//
// Created by pbustos on 5/4/21.
//

#ifndef CONTROLLER_DSR_PLAN_H
#define CONTROLLER_DSR_PLAN_H

class Plan
{
    public:
        enum class Actions {NONE, GOTO, BOUNCE, FOLLOW_PATH,FOLLOW_PEOPLE, ASK_FOR_FOLLOWING, ASK_FOR_STOP_FOLLOWING, RECOGNIZE_PEOPLE, TALKING_WITH_PEOPLE, TALKING_WITH_ROBOT, ASK_FOR_STOP_TALKING, SEARCHING_PERSON};
        enum class PlanState {INACTIVE, RUNNING, CREATING, FINISHED, READY};
        Plan()
        {
            empty = true;
            active = false;
            state = PlanState::INACTIVE;
        };
        Plan(Actions action_)
        {
            this->action = action_;
            planJ.insert(QString::fromStdString(convert_action_to_string(action_)), QVariantMap());
            state = PlanState::CREATING;
        }
        Plan(const std::string &plan_string)
        {
            try
            {
                QJsonDocument doc = QJsonDocument::fromJson(QString::fromStdString(plan_string).toUtf8());
                planJ = qvariant_cast<QVariantMap>(doc.toVariant());
                QString act = planJ.keys().front();  // OJO ES SOLO LA PRIMERA KEY DEL MAPA
                this->action = convert_string_to_action(act.toStdString());
                state = PlanState::CREATING;
            }
            catch (const std::exception &e)
            {
                std::cout << e.what() << std::endl;
                std::terminate();
            };
        };
        void new_plan(Actions action_)
        {
            planJ.clear();
            planJ.insert(convert_action_to_qstring(action_), QVariantMap());
            this->action = action_;
            state = PlanState::CREATING;
        }
        void insert_attribute(const std::string &key, QVariant value)
        {
            auto p = qvariant_cast<QVariantMap>(planJ[convert_action_to_qstring(this->action)]);
            p.insert(QString::fromStdString(key), value);
            planJ[convert_action_to_qstring(this->action)].setValue(p);
        }
        QVariant get_attribute(const std::string &key)
        {
            auto p = qvariant_cast<QVariantMap>(planJ[convert_action_to_qstring(this->action)]);
            return p.value(QString::fromStdString(key));
        }
        void reset()
        {
            planJ.clear();
            state = PlanState::INACTIVE;
            //action = Plan::Actions::NONE;
        }
        QString get_action() const { return convert_action_to_qstring(this->action); };
        std::string to_json() const
        {
            QJsonDocument json = QJsonDocument::fromVariant(planJ);
            //QTextStream ts(stdout);
            //ts << json.toJson();
            return QString(json.toJson()).toStdString();
        }
        std::string pprint() const
        {
            std::stringstream ss;
            ss << "Plan:" << std::endl;
            ss << "\t" << "Action: " << convert_action_to_string(action) << std::endl;
            for(const auto p : planJ.keys())
                for(const auto e : qvariant_cast<QVariantMap>(planJ[p]).keys())
                {
                    ss << "\t\t" << e.toStdString() << " : " << qvariant_cast<QVariantMap>(planJ[p])[e].toString().toStdString() << std::endl;
                }
           return ss.str();
        };
        bool is_running() const {return state == PlanState::RUNNING;};
        void set_running()
        {
            if(state == PlanState::READY)
                state = PlanState::RUNNING;
            else
                qWarning() << __FUNCTION__ << " Plan is not READY to start RUNNING";
        }

        bool is_valid() const { return state != PlanState::INACTIVE;};
        bool is_action(Actions test) const { return test == this->action;};
        bool is_complete()
        {
            if( state == PlanState::CREATING)
                try
                {
                    if( bool r = action_to_tests.at(action); r == true)
                    {
                      state = PlanState::READY;
                      return true;
                    }
                }
                catch(const std::exception &e){std::cout << e.what() << std::endl;};
            return false;
        }

        bool is_finished() const {return state == PlanState::FINISHED;};

    // Variables for specific plans ÑAPA
        std::vector<float> x_path, y_path;

    private:
        QVariantMap planJ;
        Actions action;
        PlanState state = PlanState::INACTIVE;

        bool empty = true;
        bool active = false;
        std::map<Actions, std::string> actions_to_strings
         {
            {Actions::GOTO, "GOTO"},
            {Actions::BOUNCE, "BOUNCE"},
            {Actions::FOLLOW_PATH, "FOLLOW_PATH"},
            {Actions::NONE, "NONE"},
            {Actions::FOLLOW_PEOPLE,"FOLLOW_PEOPLE"},
            {Actions::RECOGNIZE_PEOPLE,"RECOGNIZE_PEOPLE"},
            {Actions::TALKING_WITH_PEOPLE,"TALKING_WITH_PEOPLE"},
            {Actions::TALKING_WITH_ROBOT,"TALKING_WITH_ROBOT"},
            {Actions::ASK_FOR_FOLLOWING, "ASK_FOR_FOLLOWING"},
            {Actions::ASK_FOR_STOP_FOLLOWING, "ASK_FOR_STOP_FOLLOWING"},
            {Actions::SEARCHING_PERSON, "SEARCHING_PERSON"},
            {Actions::ASK_FOR_STOP_TALKING, "ASK_FOR_STOP_TALKING"}
         };
        std::map<std::string, Actions> strings_to_actions
        {
            {"GOTO", Actions::GOTO},
            {"BOUNCE", Actions::BOUNCE},
            {"FOLLOW_PATH", Actions::FOLLOW_PATH},
            {"NONE", Actions::NONE},
            {"FOLLOW_PEOPLE", Actions::FOLLOW_PEOPLE},
            {"RECOGNIZE_PEOPLE", Actions::RECOGNIZE_PEOPLE},
            {"TALKING_WITH_PEOPLE", Actions::TALKING_WITH_PEOPLE},
            {"TALKING_WITH_ROBOT", Actions::TALKING_WITH_ROBOT},
            {"ASK_FOR_FOLLOWING", Actions::ASK_FOR_FOLLOWING},
            {"ASK_FOR_STOP_FOLLOWING", Actions::ASK_FOR_STOP_FOLLOWING},
            {"SEARCHING_PERSON", Actions::SEARCHING_PERSON},
            {"ASK_FOR_STOP_TALKING", Actions::ASK_FOR_STOP_TALKING}
        };
        std::string convert_action_to_string(Actions action) const
        {
            try
            { return actions_to_strings.at(action);}
            catch(const std::exception &e){std::cout << e.what() << std::endl; return std::string();};
        }
        QString convert_action_to_qstring(Actions action) const
        {
            try
            { return QString::fromStdString(actions_to_strings.at(action));}
            catch(const std::exception &e){std::cout << e.what() << std::endl; return QString();};
        }
        Actions convert_string_to_action(const std::string &action) const
        {
            try
            { return strings_to_actions.at(action);}
            catch(const std::exception &e){std::cout << e.what() << std::endl; std::terminate(); };
        }
        Actions convert_qstring_to_action(const QString &action) const
        {
            try
            { return strings_to_actions.at(action.toStdString());}
            catch(const std::exception &e){std::cout << e.what() << std::endl; std::terminate();};
        }

        // Tests for specific plans
        typedef bool (Plan::*Test)();
        bool GOTO_test()
        {
            auto params = qvariant_cast<QVariantMap>(planJ["GOTO"]);
            if(not params.contains("x"))
            { qWarning() << __FUNCTION__ << "Missing x"; return false; }
            if(not params.contains("y"))
            { qWarning() << __FUNCTION__ << "Missing y"; return false; }
//            if(not params.contains("angle"))
//            { qWarning() << __FUNCTION__ << "Missing angle"; return false; }
            return true;
        };
        bool BOUNCE_test() { return true; };
        bool FOLLOW_PATH_test()
        {
            if( action != Plan::Actions::NONE)
                return true;
            else
                return false;
        };
        bool FOLLOW_PEOPLE_test() { return true; };
        bool RECOGNIZE_PEOPLE_test() { return true; };
        bool TALKING_WITH_PEOPLE_test() { return true; };
        bool TALKING_WITH_ROBOT_test() { return true; };
        bool ASK_FOR_FOLLOWING_test() { return true; };
        bool ASK_FOR_STOP_FOLLOWING_test() { return true; };
        bool SEARCHING_PERSON_test() { return true; };
        bool ASK_FOR_STOP_TALKING_test() { return true; };

        std::map <Actions, Test> action_to_tests
        {
            {Actions::GOTO, &Plan::GOTO_test},
            {Actions::BOUNCE, &Plan::BOUNCE_test},
            {Actions::FOLLOW_PATH, &Plan::FOLLOW_PATH_test},
            {Actions::FOLLOW_PEOPLE, &Plan::FOLLOW_PEOPLE_test},
            {Actions::RECOGNIZE_PEOPLE, &Plan::RECOGNIZE_PEOPLE_test},
            {Actions::TALKING_WITH_PEOPLE, &Plan::TALKING_WITH_PEOPLE_test},
            {Actions::TALKING_WITH_ROBOT, &Plan::TALKING_WITH_ROBOT_test},
            {Actions::ASK_FOR_FOLLOWING, &Plan::ASK_FOR_FOLLOWING_test},
            {Actions::ASK_FOR_STOP_FOLLOWING, &Plan::ASK_FOR_STOP_FOLLOWING_test},
            {Actions::SEARCHING_PERSON, &Plan::SEARCHING_PERSON_test},
            {Actions::ASK_FOR_STOP_TALKING, &Plan::ASK_FOR_STOP_TALKING_test}
        };


};

#endif //CONTROLLER_DSR_MISSION_H


//        Plan(const std::string &plan_string_)
//        {
//            QJsonDocument doc = QJsonDocument::fromJson(QString::fromStdString(plan_string_).toUtf8());
//            QJsonObject planJson = doc.object();
//            QJsonArray actionArray = planJson.value("plan").toArray();
//            QJsonObject action_0 = actionArray.at(0).toObject();
//            QString action_s = action_0.value("action").toString();
//            if (action_s == "GOTO")
//            {
//                QJsonObject action_params = action_0.value("params").toObject();
//                QString object = action_params.value("object").toString();
//                QJsonArray location = action_params.value("location").toArray();
//                params["x"] = location.at(0).toDouble();
//                params["y"] = location.at(1).toDouble();
//                params["angle"] = location.at(2).toDouble();
//                action = Plan::Actions::GOTO;
//                target_place = object.toStdString();
//            }
//            if (action_s == "BOUNCE")
//                action = Plan::Actions::BOUNCE;
//            if (action_s == "FOLLOW_PATH")
//                action = Plan::Actions::FOLLOW_PATH;
//
//            plan_string = plan_string_;
//            empty = false;
//        };
