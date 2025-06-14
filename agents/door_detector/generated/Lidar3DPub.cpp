//
// Copyright (c) ZeroC, Inc. All rights reserved.
//
//
// Ice version 3.7.6
//
// <auto-generated>
//
// Generated from file `Lidar3DPub.ice'
//
// Warning: do not edit this file.
//
// </auto-generated>
//

#include <Lidar3DPub.h>
#include <IceUtil/PushDisableWarnings.h>
#include <Ice/LocalException.h>
#include <Ice/ValueFactory.h>
#include <Ice/OutgoingAsync.h>
#include <Ice/InputStream.h>
#include <Ice/OutputStream.h>
#include <IceUtil/PopDisableWarnings.h>

#if defined(_MSC_VER)
#   pragma warning(disable:4458) // declaration of ... hides class member
#elif defined(__clang__)
#   pragma clang diagnostic ignored "-Wshadow"
#elif defined(__GNUC__)
#   pragma GCC diagnostic ignored "-Wshadow"
#endif

#ifndef ICE_IGNORE_VERSION
#   if ICE_INT_VERSION / 100 != 307
#       error Ice version mismatch!
#   endif
#   if ICE_INT_VERSION % 100 >= 50
#       error Beta header file detected
#   endif
#   if ICE_INT_VERSION % 100 < 6
#       error Ice patch level mismatch!
#   endif
#endif

#ifdef ICE_CPP11_MAPPING // C++11 mapping

namespace
{

const ::std::string iceC_RoboCompLidar3DPub_Lidar3DPub_ids[2] =
{
    "::Ice::Object",
    "::RoboCompLidar3DPub::Lidar3DPub"
};
const ::std::string iceC_RoboCompLidar3DPub_Lidar3DPub_ops[] =
{
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping",
    "pushLidarData"
};
const ::std::string iceC_RoboCompLidar3DPub_Lidar3DPub_pushLidarData_name = "pushLidarData";

}

bool
RoboCompLidar3DPub::Lidar3DPub::ice_isA(::std::string s, const ::Ice::Current&) const
{
    return ::std::binary_search(iceC_RoboCompLidar3DPub_Lidar3DPub_ids, iceC_RoboCompLidar3DPub_Lidar3DPub_ids + 2, s);
}

::std::vector<::std::string>
RoboCompLidar3DPub::Lidar3DPub::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector<::std::string>(&iceC_RoboCompLidar3DPub_Lidar3DPub_ids[0], &iceC_RoboCompLidar3DPub_Lidar3DPub_ids[2]);
}

::std::string
RoboCompLidar3DPub::Lidar3DPub::ice_id(const ::Ice::Current&) const
{
    return ice_staticId();
}

const ::std::string&
RoboCompLidar3DPub::Lidar3DPub::ice_staticId()
{
    static const ::std::string typeId = "::RoboCompLidar3DPub::Lidar3DPub";
    return typeId;
}

/// \cond INTERNAL
bool
RoboCompLidar3DPub::Lidar3DPub::_iceD_pushLidarData(::IceInternal::Incoming& inS, const ::Ice::Current& current)
{
    _iceCheckMode(::Ice::OperationMode::Idempotent, current.mode);
    auto istr = inS.startReadParams();
    ::RoboCompLidar3D::TDataCategory iceP_lidarData;
    istr->readAll(iceP_lidarData);
    inS.endReadParams();
    this->pushLidarData(::std::move(iceP_lidarData), current);
    inS.writeEmptyParams();
    return true;
}
/// \endcond

/// \cond INTERNAL
bool
RoboCompLidar3DPub::Lidar3DPub::_iceDispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair<const ::std::string*, const ::std::string*> r = ::std::equal_range(iceC_RoboCompLidar3DPub_Lidar3DPub_ops, iceC_RoboCompLidar3DPub_Lidar3DPub_ops + 5, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - iceC_RoboCompLidar3DPub_Lidar3DPub_ops)
    {
        case 0:
        {
            return _iceD_ice_id(in, current);
        }
        case 1:
        {
            return _iceD_ice_ids(in, current);
        }
        case 2:
        {
            return _iceD_ice_isA(in, current);
        }
        case 3:
        {
            return _iceD_ice_ping(in, current);
        }
        case 4:
        {
            return _iceD_pushLidarData(in, current);
        }
        default:
        {
            assert(false);
            throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
        }
    }
}
/// \endcond

/// \cond INTERNAL
void
RoboCompLidar3DPub::Lidar3DPubPrx::_iceI_pushLidarData(const ::std::shared_ptr<::IceInternal::OutgoingAsyncT<void>>& outAsync, const ::RoboCompLidar3D::TDataCategory& iceP_lidarData, const ::Ice::Context& context)
{
    outAsync->invoke(iceC_RoboCompLidar3DPub_Lidar3DPub_pushLidarData_name, ::Ice::OperationMode::Idempotent, ::Ice::FormatType::DefaultFormat, context,
        [&](::Ice::OutputStream* ostr)
        {
            ostr->writeAll(iceP_lidarData);
        },
        nullptr);
}
/// \endcond

/// \cond INTERNAL
::std::shared_ptr<::Ice::ObjectPrx>
RoboCompLidar3DPub::Lidar3DPubPrx::_newInstance() const
{
    return ::IceInternal::createProxy<Lidar3DPubPrx>();
}
/// \endcond

const ::std::string&
RoboCompLidar3DPub::Lidar3DPubPrx::ice_staticId()
{
    return Lidar3DPub::ice_staticId();
}

#else // C++98 mapping

namespace
{

const ::std::string iceC_RoboCompLidar3DPub_Lidar3DPub_pushLidarData_name = "pushLidarData";

}

/// \cond INTERNAL
::IceProxy::Ice::Object* ::IceProxy::RoboCompLidar3DPub::upCast(Lidar3DPub* p) { return p; }

void
::IceProxy::RoboCompLidar3DPub::_readProxy(::Ice::InputStream* istr, ::IceInternal::ProxyHandle< Lidar3DPub>& v)
{
    ::Ice::ObjectPrx proxy;
    istr->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new Lidar3DPub;
        v->_copyFrom(proxy);
    }
}
/// \endcond

::Ice::AsyncResultPtr
IceProxy::RoboCompLidar3DPub::Lidar3DPub::_iceI_begin_pushLidarData(const ::RoboCompLidar3D::TDataCategory& iceP_lidarData, const ::Ice::Context& context, const ::IceInternal::CallbackBasePtr& del, const ::Ice::LocalObjectPtr& cookie, bool sync)
{
    ::IceInternal::OutgoingAsyncPtr result = new ::IceInternal::CallbackOutgoing(this, iceC_RoboCompLidar3DPub_Lidar3DPub_pushLidarData_name, del, cookie, sync);
    try
    {
        result->prepare(iceC_RoboCompLidar3DPub_Lidar3DPub_pushLidarData_name, ::Ice::Idempotent, context);
        ::Ice::OutputStream* ostr = result->startWriteParams(::Ice::DefaultFormat);
        ostr->write(iceP_lidarData);
        result->endWriteParams();
        result->invoke(iceC_RoboCompLidar3DPub_Lidar3DPub_pushLidarData_name);
    }
    catch(const ::Ice::Exception& ex)
    {
        result->abort(ex);
    }
    return result;
}

void
IceProxy::RoboCompLidar3DPub::Lidar3DPub::end_pushLidarData(const ::Ice::AsyncResultPtr& result)
{
    _end(result, iceC_RoboCompLidar3DPub_Lidar3DPub_pushLidarData_name);
}

/// \cond INTERNAL
::IceProxy::Ice::Object*
IceProxy::RoboCompLidar3DPub::Lidar3DPub::_newInstance() const
{
    return new Lidar3DPub;
}
/// \endcond

const ::std::string&
IceProxy::RoboCompLidar3DPub::Lidar3DPub::ice_staticId()
{
    return ::RoboCompLidar3DPub::Lidar3DPub::ice_staticId();
}

RoboCompLidar3DPub::Lidar3DPub::~Lidar3DPub()
{
}

/// \cond INTERNAL
::Ice::Object* RoboCompLidar3DPub::upCast(Lidar3DPub* p) { return p; }

/// \endcond

namespace
{
const ::std::string iceC_RoboCompLidar3DPub_Lidar3DPub_ids[2] =
{
    "::Ice::Object",
    "::RoboCompLidar3DPub::Lidar3DPub"
};

}

bool
RoboCompLidar3DPub::Lidar3DPub::ice_isA(const ::std::string& s, const ::Ice::Current&) const
{
    return ::std::binary_search(iceC_RoboCompLidar3DPub_Lidar3DPub_ids, iceC_RoboCompLidar3DPub_Lidar3DPub_ids + 2, s);
}

::std::vector< ::std::string>
RoboCompLidar3DPub::Lidar3DPub::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&iceC_RoboCompLidar3DPub_Lidar3DPub_ids[0], &iceC_RoboCompLidar3DPub_Lidar3DPub_ids[2]);
}

const ::std::string&
RoboCompLidar3DPub::Lidar3DPub::ice_id(const ::Ice::Current&) const
{
    return ice_staticId();
}

const ::std::string&
RoboCompLidar3DPub::Lidar3DPub::ice_staticId()
{
#ifdef ICE_HAS_THREAD_SAFE_LOCAL_STATIC
    static const ::std::string typeId = "::RoboCompLidar3DPub::Lidar3DPub";
    return typeId;
#else
    return iceC_RoboCompLidar3DPub_Lidar3DPub_ids[1];
#endif
}

/// \cond INTERNAL
bool
RoboCompLidar3DPub::Lidar3DPub::_iceD_pushLidarData(::IceInternal::Incoming& inS, const ::Ice::Current& current)
{
    _iceCheckMode(::Ice::Idempotent, current.mode);
    ::Ice::InputStream* istr = inS.startReadParams();
    ::RoboCompLidar3D::TDataCategory iceP_lidarData;
    istr->read(iceP_lidarData);
    inS.endReadParams();
    this->pushLidarData(iceP_lidarData, current);
    inS.writeEmptyParams();
    return true;
}
/// \endcond

namespace
{
const ::std::string iceC_RoboCompLidar3DPub_Lidar3DPub_all[] =
{
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping",
    "pushLidarData"
};

}

/// \cond INTERNAL
bool
RoboCompLidar3DPub::Lidar3DPub::_iceDispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair<const ::std::string*, const ::std::string*> r = ::std::equal_range(iceC_RoboCompLidar3DPub_Lidar3DPub_all, iceC_RoboCompLidar3DPub_Lidar3DPub_all + 5, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - iceC_RoboCompLidar3DPub_Lidar3DPub_all)
    {
        case 0:
        {
            return _iceD_ice_id(in, current);
        }
        case 1:
        {
            return _iceD_ice_ids(in, current);
        }
        case 2:
        {
            return _iceD_ice_isA(in, current);
        }
        case 3:
        {
            return _iceD_ice_ping(in, current);
        }
        case 4:
        {
            return _iceD_pushLidarData(in, current);
        }
        default:
        {
            assert(false);
            throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
        }
    }
}
/// \endcond

/// \cond STREAM
void
RoboCompLidar3DPub::Lidar3DPub::_iceWriteImpl(::Ice::OutputStream* ostr) const
{
    ostr->startSlice(ice_staticId(), -1, true);
    ::Ice::StreamWriter< Lidar3DPub, ::Ice::OutputStream>::write(ostr, *this);
    ostr->endSlice();
}

void
RoboCompLidar3DPub::Lidar3DPub::_iceReadImpl(::Ice::InputStream* istr)
{
    istr->startSlice();
    ::Ice::StreamReader< Lidar3DPub, ::Ice::InputStream>::read(istr, *this);
    istr->endSlice();
}
/// \endcond

/// \cond INTERNAL
void
RoboCompLidar3DPub::_icePatchObjectPtr(Lidar3DPubPtr& handle, const ::Ice::ObjectPtr& v)
{
    handle = Lidar3DPubPtr::dynamicCast(v);
    if(v && !handle)
    {
        IceInternal::Ex::throwUOE(Lidar3DPub::ice_staticId(), v);
    }
}
/// \endcond

namespace Ice
{
}

#endif
