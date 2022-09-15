/*
 *    Copyright (C) 2022 by YOUR NAME HERE
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
#include "agenteconversacionalI.h"

AgenteConversacionalI::AgenteConversacionalI(GenericWorker *_worker)
{
	worker = _worker;
}


AgenteConversacionalI::~AgenteConversacionalI()
{
}


void AgenteConversacionalI::asynchronousIntentionReceiver(int intention, const Ice::Current&)
{
	worker->AgenteConversacional_asynchronousIntentionReceiver(intention);
}

void AgenteConversacionalI::componentState(int state, const Ice::Current&)
{
	worker->AgenteConversacional_componentState(state);
}

int AgenteConversacionalI::situationChecking(const Ice::Current&)
{
	return worker->AgenteConversacional_situationChecking();
}

