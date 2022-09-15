#
#    Copyright (C) 2022 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

import sys, os, Ice

ROBOCOMP = ''
try:
    ROBOCOMP = os.environ['ROBOCOMP']
except:
    print('$ROBOCOMP environment variable not set, using the default value /opt/robocomp')
    ROBOCOMP = '/opt/robocomp'
if len(ROBOCOMP)<1:
    raise RuntimeError('ROBOCOMP environment variable not set! Exiting.')


Ice.loadSlice("-I ./src/ --all ./src/Conversation.ice")

from RoboCompConversation import *

class ConversationI(Conversation):
    def __init__(self, worker):
        self.worker = worker


    def following(self, name, role, c):
        return self.worker.Conversation_following(name, role)

    def isBlocked(self, blocked, c):
        return self.worker.Conversation_isBlocked(blocked)

    def isFollowing(self, following, c):
        return self.worker.Conversation_isFollowing(following)

    def isTalking(self, talking, c):
        return self.worker.Conversation_isTalking(talking)

    def listenToHuman(self, c):
        return self.worker.Conversation_listenToHuman()

    def lost(self, name, role, c):
        return self.worker.Conversation_lost(name, role)

    def sayHi(self, name, role, c):
        return self.worker.Conversation_sayHi(name, role)

    def saySomething(self, name, phrase, c):
        return self.worker.Conversation_saySomething(name, phrase)

    def stopFollowing(self, name, role, c):
        return self.worker.Conversation_stopFollowing(name, role)

    def talking(self, name, role, conversation, c):
        return self.worker.Conversation_talking(name, role, conversation)

    def waiting(self, name, role, c):
        return self.worker.Conversation_waiting(name, role)
