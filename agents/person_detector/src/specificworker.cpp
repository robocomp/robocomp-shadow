/*
 *    Copyright (C) 2025 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "specificworker.h"

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check) : GenericWorker(configLoader, tprx)
{
	this->startup_check_flag = startup_check;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		#ifdef HIBERNATION_ENABLED
			hibernationChecker.start(500);
		#endif

		statemachine.setChildMode(QState::ExclusiveStates);
		statemachine.start();

		auto error = statemachine.errorString();
		if (error.length() > 0){
			qWarning() << error;
			throw error;
		}
	}
}

SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
	//G->write_to_json_file("./"+agent_name+".json");
}


void SpecificWorker::initialize()
{
    std::cout << "initialize worker" << std::endl;
}


void  SpecificWorker::compute()
{
	std::cout << "Compute worker" << std::endl;
	RoboCompVisualElementsPub::TObjects processing;
	{
		std::lock_guard<std::mutex> lock(process_mutex);
		if (toProcess.empty()) return;
		
		// Usamos move semantics para transferir la propiedad
		processing = std::move(toProcess);
		toProcess.clear(); // Limpiamos el vector original
	}
	// Procesamos los objetos
	for (auto& obj : processing) 
	{
		if (obj.type == 0) // Es una persona
		{
			processPerson(obj);
		}
	}
}

void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
    //emergencyCODE
    //
    //if (SUCCESSFUL) //The componet is safe for continue
    //  emmit goToRestore()
}


//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
    //restoreCODE
    //Restore emergency component

}


int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}
void SpecificWorker::processPerson(const RoboCompVisualElementsPub::TObject& person)
{
	auto it = visual_to_dsr_id.find(person.id);
	std::cout << person.id << std::endl;
	if (it != visual_to_dsr_id.end()) 
	{
		updateDSRPerson(it->second, person.attributes);
	} 
	else 
	{
		createDSRPerson(person.id, person.attributes);
	}
}

void SpecificWorker::createDSRPerson(int visual_id, const RoboCompVisualElementsPub::TAttributes& attrs)
{
	float confidence = 0.0f;
	if (auto it = attrs.find("score"); it != attrs.end()) {
		confidence = std::stof(it->second);
	} else {
		std::cerr << "Attribute 'score' not found" << std::endl;
		return;
	}

	if (confidence > 0.75) {
		try {
			DSR::Node  node = DSR::Node::create<person_node_type>("person_"+visual_id);

			G->add_or_modify_attrib_local<parent_att>(node, (std::uint64_t)200);
			G->add_or_modify_attrib_local<pos_x_att>(node, (float)(rand()%(170)));
			G->add_or_modify_attrib_local<pos_y_att>(node, (float)(rand()%(170)));
			G->add_or_modify_attrib_local<person_id_att>(node, visual_id);

			processPersonAttributes(node, attrs);

			G->insert_node(node);
			visual_to_dsr_id[visual_id] = node.id();

			std::cout << "Creado nodo DSR para persona " << visual_id 
					<< " con ID: " << node.id() << std::endl;
		}
		catch (const std::exception &e) {
			std::cerr << "Error al crear nodo persona: " << e.what() << std::endl;
		}
	}
}

void SpecificWorker::updateDSRPerson(int visual_id, const RoboCompVisualElementsPub::TAttributes& attrs)
{
	try {
		int node_id = visual_to_dsr_id.at(visual_id);
		// Obtener el nodo existente
		std::optional<DSR::Node> node = G->get_node(node_id);
		if (!node.has_value()) {
			std::cerr << "Nodo no encontrado, ID: " << node_id << std::endl;
			createDSRPerson(visual_id, attrs);
			return;
		}
		DSR::Node node_value = node.value();

		// Procesar atributos
		processPersonAttributes(node_value, attrs);

		// Actualizar el nodo
		G->update_node(node_value);

		std::cout << "Actualizado nodo DSR para persona con ID: " << node_id << std::endl;
	}
	catch (const std::exception &e) {
		std::cerr << "Error al actualizar nodo persona: " << e.what() << std::endl;
	}
}

void SpecificWorker::processPersonAttributes(DSR::Node &node, const RoboCompVisualElementsPub::TAttributes& attrs)
{

	try
	{


        if (auto it = attrs.find("x_pos"); it != attrs.end()) {
            G->add_or_modify_attrib_local<person_pixel_x_att>(node, std::stoi(it->second));
        }
        
        // Procesar y_pos si existe
        if (auto it = attrs.find("y_pos"); it != attrs.end()) {
            G->add_or_modify_attrib_local<person_pixel_y_att>(node, std::stoi(it->second));
        }
	}
	catch (const std::exception &e) {
		std::cerr << "Error procesando atributo " << e.what() << std::endl;
	}
}

//SUBSCRIPTION to setVisualObjects method from VisualElementsPub interface
void SpecificWorker::VisualElementsPub_setVisualObjects(RoboCompVisualElementsPub::TData data)
{
	std::lock_guard<std::mutex> lock(process_mutex);
	if (this->toProcess.empty()) {
		// Usamos move semantics para evitar copias
		this->toProcess = std::move(data.objects);
	} else {
		// Reservamos espacio de antemano para evitar múltiples reallocations
		this->toProcess.reserve(this->toProcess.size() + data.objects.size());
		// Movemos los elementos uno por uno
		for (auto& obj : data.objects) {
			this->toProcess.emplace_back(std::move(obj));
		}
	}
}



/**************************************/
// From the RoboCompVisualElementsPub you can use this types:
// RoboCompVisualElementsPub::TObject
// RoboCompVisualElementsPub::TData

