//
// Copyright (c) ZeroC, Inc. All rights reserved.
//
//
// Ice version 3.7.6
//
// <auto-generated>
//
// Generated from file `SegmentatorTrackingPub.ice'
//
// Warning: do not edit this file.
//
// </auto-generated>
//

#include <SegmentatorTrackingPub.h>
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

const ::std::string iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ids[2] =
{
    "::Ice::Object",
    "::RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub"
};
const ::std::string iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ops[] =
{
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping",
    "setTrack"
};
const ::std::string iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_setTrack_name = "setTrack";

}

bool
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::ice_isA(::std::string s, const ::Ice::Current&) const
{
    return ::std::binary_search(iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ids, iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ids + 2, s);
}

::std::vector<::std::string>
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector<::std::string>(&iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ids[0], &iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ids[2]);
}

::std::string
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::ice_id(const ::Ice::Current&) const
{
    return ice_staticId();
}

const ::std::string&
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::ice_staticId()
{
    static const ::std::string typeId = "::RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub";
    return typeId;
}

/// \cond INTERNAL
bool
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::_iceD_setTrack(::IceInternal::Incoming& inS, const ::Ice::Current& current)
{
    _iceCheckMode(::Ice::OperationMode::Idempotent, current.mode);
    auto istr = inS.startReadParams();
    ::RoboCompVisualElementsPub::TObject iceP_target;
    istr->readAll(iceP_target);
    inS.endReadParams();
    this->setTrack(::std::move(iceP_target), current);
    inS.writeEmptyParams();
    return true;
}
/// \endcond

/// \cond INTERNAL
bool
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::_iceDispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair<const ::std::string*, const ::std::string*> r = ::std::equal_range(iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ops, iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ops + 5, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ops)
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
            return _iceD_setTrack(in, current);
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
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPubPrx::_iceI_setTrack(const ::std::shared_ptr<::IceInternal::OutgoingAsyncT<void>>& outAsync, const ::RoboCompVisualElementsPub::TObject& iceP_target, const ::Ice::Context& context)
{
    outAsync->invoke(iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_setTrack_name, ::Ice::OperationMode::Idempotent, ::Ice::FormatType::DefaultFormat, context,
        [&](::Ice::OutputStream* ostr)
        {
            ostr->writeAll(iceP_target);
        },
        nullptr);
}
/// \endcond

/// \cond INTERNAL
::std::shared_ptr<::Ice::ObjectPrx>
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPubPrx::_newInstance() const
{
    return ::IceInternal::createProxy<SegmentatorTrackingPubPrx>();
}
/// \endcond

const ::std::string&
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPubPrx::ice_staticId()
{
    return SegmentatorTrackingPub::ice_staticId();
}

#else // C++98 mapping

namespace
{

const ::std::string iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_setTrack_name = "setTrack";

}

/// \cond INTERNAL
::IceProxy::Ice::Object* ::IceProxy::RoboCompSegmentatorTrackingPub::upCast(SegmentatorTrackingPub* p) { return p; }

void
::IceProxy::RoboCompSegmentatorTrackingPub::_readProxy(::Ice::InputStream* istr, ::IceInternal::ProxyHandle< SegmentatorTrackingPub>& v)
{
    ::Ice::ObjectPrx proxy;
    istr->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new SegmentatorTrackingPub;
        v->_copyFrom(proxy);
    }
}
/// \endcond

::Ice::AsyncResultPtr
IceProxy::RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::_iceI_begin_setTrack(const ::RoboCompVisualElementsPub::TObject& iceP_target, const ::Ice::Context& context, const ::IceInternal::CallbackBasePtr& del, const ::Ice::LocalObjectPtr& cookie, bool sync)
{
    ::IceInternal::OutgoingAsyncPtr result = new ::IceInternal::CallbackOutgoing(this, iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_setTrack_name, del, cookie, sync);
    try
    {
        result->prepare(iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_setTrack_name, ::Ice::Idempotent, context);
        ::Ice::OutputStream* ostr = result->startWriteParams(::Ice::DefaultFormat);
        ostr->write(iceP_target);
        result->endWriteParams();
        result->invoke(iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_setTrack_name);
    }
    catch(const ::Ice::Exception& ex)
    {
        result->abort(ex);
    }
    return result;
}

void
IceProxy::RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::end_setTrack(const ::Ice::AsyncResultPtr& result)
{
    _end(result, iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_setTrack_name);
}

/// \cond INTERNAL
::IceProxy::Ice::Object*
IceProxy::RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::_newInstance() const
{
    return new SegmentatorTrackingPub;
}
/// \endcond

const ::std::string&
IceProxy::RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::ice_staticId()
{
    return ::RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::ice_staticId();
}

RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::~SegmentatorTrackingPub()
{
}

/// \cond INTERNAL
::Ice::Object* RoboCompSegmentatorTrackingPub::upCast(SegmentatorTrackingPub* p) { return p; }

/// \endcond

namespace
{
const ::std::string iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ids[2] =
{
    "::Ice::Object",
    "::RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub"
};

}

bool
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::ice_isA(const ::std::string& s, const ::Ice::Current&) const
{
    return ::std::binary_search(iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ids, iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ids + 2, s);
}

::std::vector< ::std::string>
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ids[0], &iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ids[2]);
}

const ::std::string&
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::ice_id(const ::Ice::Current&) const
{
    return ice_staticId();
}

const ::std::string&
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::ice_staticId()
{
#ifdef ICE_HAS_THREAD_SAFE_LOCAL_STATIC
    static const ::std::string typeId = "::RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub";
    return typeId;
#else
    return iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_ids[1];
#endif
}

/// \cond INTERNAL
bool
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::_iceD_setTrack(::IceInternal::Incoming& inS, const ::Ice::Current& current)
{
    _iceCheckMode(::Ice::Idempotent, current.mode);
    ::Ice::InputStream* istr = inS.startReadParams();
    ::RoboCompVisualElementsPub::TObject iceP_target;
    istr->read(iceP_target);
    inS.endReadParams();
    this->setTrack(iceP_target, current);
    inS.writeEmptyParams();
    return true;
}
/// \endcond

namespace
{
const ::std::string iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_all[] =
{
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping",
    "setTrack"
};

}

/// \cond INTERNAL
bool
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::_iceDispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair<const ::std::string*, const ::std::string*> r = ::std::equal_range(iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_all, iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_all + 5, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - iceC_RoboCompSegmentatorTrackingPub_SegmentatorTrackingPub_all)
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
            return _iceD_setTrack(in, current);
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
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::_iceWriteImpl(::Ice::OutputStream* ostr) const
{
    ostr->startSlice(ice_staticId(), -1, true);
    ::Ice::StreamWriter< SegmentatorTrackingPub, ::Ice::OutputStream>::write(ostr, *this);
    ostr->endSlice();
}

void
RoboCompSegmentatorTrackingPub::SegmentatorTrackingPub::_iceReadImpl(::Ice::InputStream* istr)
{
    istr->startSlice();
    ::Ice::StreamReader< SegmentatorTrackingPub, ::Ice::InputStream>::read(istr, *this);
    istr->endSlice();
}
/// \endcond

/// \cond INTERNAL
void
RoboCompSegmentatorTrackingPub::_icePatchObjectPtr(SegmentatorTrackingPubPtr& handle, const ::Ice::ObjectPtr& v)
{
    handle = SegmentatorTrackingPubPtr::dynamicCast(v);
    if(v && !handle)
    {
        IceInternal::Ex::throwUOE(SegmentatorTrackingPub::ice_staticId(), v);
    }
}
/// \endcond

namespace Ice
{
}

#endif