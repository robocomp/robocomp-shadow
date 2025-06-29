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

#ifndef __Lidar3DPub_h__
#define __Lidar3DPub_h__

#include <IceUtil/PushDisableWarnings.h>
#include <Ice/ProxyF.h>
#include <Ice/ObjectF.h>
#include <Ice/ValueF.h>
#include <Ice/Exception.h>
#include <Ice/LocalObject.h>
#include <Ice/StreamHelpers.h>
#include <Ice/Comparable.h>
#include <Ice/Proxy.h>
#include <Ice/Object.h>
#include <Ice/GCObject.h>
#include <Ice/Value.h>
#include <Ice/Incoming.h>
#include <Ice/FactoryTableInit.h>
#include <IceUtil/ScopedArray.h>
#include <Ice/Optional.h>
#include <Lidar3D.h>
#include <IceUtil/UndefSysMacros.h>

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

namespace RoboCompLidar3DPub
{

class Lidar3DPub;
class Lidar3DPubPrx;

}

namespace RoboCompLidar3DPub
{

class Lidar3DPub : public virtual ::Ice::Object
{
public:

    using ProxyType = Lidar3DPubPrx;

    /**
     * Determines whether this object supports an interface with the given Slice type ID.
     * @param id The fully-scoped Slice type ID.
     * @param current The Current object for the invocation.
     * @return True if this object supports the interface, false, otherwise.
     */
    virtual bool ice_isA(::std::string id, const ::Ice::Current& current) const override;

    /**
     * Obtains a list of the Slice type IDs representing the interfaces supported by this object.
     * @param current The Current object for the invocation.
     * @return A list of fully-scoped type IDs.
     */
    virtual ::std::vector<::std::string> ice_ids(const ::Ice::Current& current) const override;

    /**
     * Obtains a Slice type ID representing the most-derived interface supported by this object.
     * @param current The Current object for the invocation.
     * @return A fully-scoped type ID.
     */
    virtual ::std::string ice_id(const ::Ice::Current& current) const override;

    /**
     * Obtains the Slice type ID corresponding to this class.
     * @return A fully-scoped type ID.
     */
    static const ::std::string& ice_staticId();

    virtual void pushLidarData(::RoboCompLidar3D::TDataCategory lidarData, const ::Ice::Current& current) = 0;
    /// \cond INTERNAL
    bool _iceD_pushLidarData(::IceInternal::Incoming&, const ::Ice::Current&);
    /// \endcond

    /// \cond INTERNAL
    virtual bool _iceDispatch(::IceInternal::Incoming&, const ::Ice::Current&) override;
    /// \endcond
};

}

namespace RoboCompLidar3DPub
{

class Lidar3DPubPrx : public virtual ::Ice::Proxy<Lidar3DPubPrx, ::Ice::ObjectPrx>
{
public:

    void pushLidarData(const ::RoboCompLidar3D::TDataCategory& lidarData, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        _makePromiseOutgoing<void>(true, this, &Lidar3DPubPrx::_iceI_pushLidarData, lidarData, context).get();
    }

    template<template<typename> class P = ::std::promise>
    auto pushLidarDataAsync(const ::RoboCompLidar3D::TDataCategory& lidarData, const ::Ice::Context& context = ::Ice::noExplicitContext)
        -> decltype(::std::declval<P<void>>().get_future())
    {
        return _makePromiseOutgoing<void, P>(false, this, &Lidar3DPubPrx::_iceI_pushLidarData, lidarData, context);
    }

    ::std::function<void()>
    pushLidarDataAsync(const ::RoboCompLidar3D::TDataCategory& lidarData,
                       ::std::function<void()> response,
                       ::std::function<void(::std::exception_ptr)> ex = nullptr,
                       ::std::function<void(bool)> sent = nullptr,
                       const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        return _makeLamdaOutgoing<void>(std::move(response), std::move(ex), std::move(sent), this, &RoboCompLidar3DPub::Lidar3DPubPrx::_iceI_pushLidarData, lidarData, context);
    }

    /// \cond INTERNAL
    void _iceI_pushLidarData(const ::std::shared_ptr<::IceInternal::OutgoingAsyncT<void>>&, const ::RoboCompLidar3D::TDataCategory&, const ::Ice::Context&);
    /// \endcond

    /**
     * Obtains the Slice type ID of this interface.
     * @return The fully-scoped type ID.
     */
    static const ::std::string& ice_staticId();

protected:

    /// \cond INTERNAL
    Lidar3DPubPrx() = default;
    friend ::std::shared_ptr<Lidar3DPubPrx> IceInternal::createProxy<Lidar3DPubPrx>();

    virtual ::std::shared_ptr<::Ice::ObjectPrx> _newInstance() const override;
    /// \endcond
};

}

/// \cond STREAM
namespace Ice
{

}
/// \endcond

/// \cond INTERNAL
namespace RoboCompLidar3DPub
{

using Lidar3DPubPtr = ::std::shared_ptr<Lidar3DPub>;
using Lidar3DPubPrxPtr = ::std::shared_ptr<Lidar3DPubPrx>;

}
/// \endcond

#else // C++98 mapping

namespace IceProxy
{

namespace RoboCompLidar3DPub
{

class Lidar3DPub;
/// \cond INTERNAL
void _readProxy(::Ice::InputStream*, ::IceInternal::ProxyHandle< Lidar3DPub>&);
::IceProxy::Ice::Object* upCast(Lidar3DPub*);
/// \endcond

}

}

namespace RoboCompLidar3DPub
{

class Lidar3DPub;
/// \cond INTERNAL
::Ice::Object* upCast(Lidar3DPub*);
/// \endcond
typedef ::IceInternal::Handle< Lidar3DPub> Lidar3DPubPtr;
typedef ::IceInternal::ProxyHandle< ::IceProxy::RoboCompLidar3DPub::Lidar3DPub> Lidar3DPubPrx;
typedef Lidar3DPubPrx Lidar3DPubPrxPtr;
/// \cond INTERNAL
void _icePatchObjectPtr(Lidar3DPubPtr&, const ::Ice::ObjectPtr&);
/// \endcond

}

namespace RoboCompLidar3DPub
{

/**
 * Base class for asynchronous callback wrapper classes used for calls to
 * IceProxy::RoboCompLidar3DPub::Lidar3DPub::begin_pushLidarData.
 * Create a wrapper instance by calling ::RoboCompLidar3DPub::newCallback_Lidar3DPub_pushLidarData.
 */
class Callback_Lidar3DPub_pushLidarData_Base : public virtual ::IceInternal::CallbackBase { };
typedef ::IceUtil::Handle< Callback_Lidar3DPub_pushLidarData_Base> Callback_Lidar3DPub_pushLidarDataPtr;

}

namespace IceProxy
{

namespace RoboCompLidar3DPub
{

class Lidar3DPub : public virtual ::Ice::Proxy<Lidar3DPub, ::IceProxy::Ice::Object>
{
public:

    void pushLidarData(const ::RoboCompLidar3D::TDataCategory& lidarData, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        end_pushLidarData(_iceI_begin_pushLidarData(lidarData, context, ::IceInternal::dummyCallback, 0, true));
    }

    ::Ice::AsyncResultPtr begin_pushLidarData(const ::RoboCompLidar3D::TDataCategory& lidarData, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        return _iceI_begin_pushLidarData(lidarData, context, ::IceInternal::dummyCallback, 0);
    }

    ::Ice::AsyncResultPtr begin_pushLidarData(const ::RoboCompLidar3D::TDataCategory& lidarData, const ::Ice::CallbackPtr& cb, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_pushLidarData(lidarData, ::Ice::noExplicitContext, cb, cookie);
    }

    ::Ice::AsyncResultPtr begin_pushLidarData(const ::RoboCompLidar3D::TDataCategory& lidarData, const ::Ice::Context& context, const ::Ice::CallbackPtr& cb, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_pushLidarData(lidarData, context, cb, cookie);
    }

    ::Ice::AsyncResultPtr begin_pushLidarData(const ::RoboCompLidar3D::TDataCategory& lidarData, const ::RoboCompLidar3DPub::Callback_Lidar3DPub_pushLidarDataPtr& cb, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_pushLidarData(lidarData, ::Ice::noExplicitContext, cb, cookie);
    }

    ::Ice::AsyncResultPtr begin_pushLidarData(const ::RoboCompLidar3D::TDataCategory& lidarData, const ::Ice::Context& context, const ::RoboCompLidar3DPub::Callback_Lidar3DPub_pushLidarDataPtr& cb, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_pushLidarData(lidarData, context, cb, cookie);
    }

    void end_pushLidarData(const ::Ice::AsyncResultPtr& result);

private:

    ::Ice::AsyncResultPtr _iceI_begin_pushLidarData(const ::RoboCompLidar3D::TDataCategory&, const ::Ice::Context&, const ::IceInternal::CallbackBasePtr&, const ::Ice::LocalObjectPtr& cookie = 0, bool sync = false);

public:

    /**
     * Obtains the Slice type ID corresponding to this interface.
     * @return A fully-scoped type ID.
     */
    static const ::std::string& ice_staticId();

protected:
    /// \cond INTERNAL

    virtual ::IceProxy::Ice::Object* _newInstance() const;
    /// \endcond
};

}

}

namespace RoboCompLidar3DPub
{

class Lidar3DPub : public virtual ::Ice::Object
{
public:

    typedef Lidar3DPubPrx ProxyType;
    typedef Lidar3DPubPtr PointerType;

    virtual ~Lidar3DPub();

#ifdef ICE_CPP11_COMPILER
    Lidar3DPub() = default;
    Lidar3DPub(const Lidar3DPub&) = default;
    Lidar3DPub& operator=(const Lidar3DPub&) = default;
#endif

    /**
     * Determines whether this object supports an interface with the given Slice type ID.
     * @param id The fully-scoped Slice type ID.
     * @param current The Current object for the invocation.
     * @return True if this object supports the interface, false, otherwise.
     */
    virtual bool ice_isA(const ::std::string& id, const ::Ice::Current& current = ::Ice::emptyCurrent) const;

    /**
     * Obtains a list of the Slice type IDs representing the interfaces supported by this object.
     * @param current The Current object for the invocation.
     * @return A list of fully-scoped type IDs.
     */
    virtual ::std::vector< ::std::string> ice_ids(const ::Ice::Current& current = ::Ice::emptyCurrent) const;

    /**
     * Obtains a Slice type ID representing the most-derived interface supported by this object.
     * @param current The Current object for the invocation.
     * @return A fully-scoped type ID.
     */
    virtual const ::std::string& ice_id(const ::Ice::Current& current = ::Ice::emptyCurrent) const;

    /**
     * Obtains the Slice type ID corresponding to this class.
     * @return A fully-scoped type ID.
     */
    static const ::std::string& ice_staticId();

    virtual void pushLidarData(const ::RoboCompLidar3D::TDataCategory& lidarData, const ::Ice::Current& current = ::Ice::emptyCurrent) = 0;
    /// \cond INTERNAL
    bool _iceD_pushLidarData(::IceInternal::Incoming&, const ::Ice::Current&);
    /// \endcond

    /// \cond INTERNAL
    virtual bool _iceDispatch(::IceInternal::Incoming&, const ::Ice::Current&);
    /// \endcond

protected:

    /// \cond STREAM
    virtual void _iceWriteImpl(::Ice::OutputStream*) const;
    virtual void _iceReadImpl(::Ice::InputStream*);
    /// \endcond
};

/// \cond INTERNAL
inline bool operator==(const Lidar3DPub& lhs, const Lidar3DPub& rhs)
{
    return static_cast<const ::Ice::Object&>(lhs) == static_cast<const ::Ice::Object&>(rhs);
}

inline bool operator<(const Lidar3DPub& lhs, const Lidar3DPub& rhs)
{
    return static_cast<const ::Ice::Object&>(lhs) < static_cast<const ::Ice::Object&>(rhs);
}
/// \endcond

}

/// \cond STREAM
namespace Ice
{

}
/// \endcond

namespace RoboCompLidar3DPub
{

/**
 * Type-safe asynchronous callback wrapper class used for calls to
 * IceProxy::RoboCompLidar3DPub::Lidar3DPub::begin_pushLidarData.
 * Create a wrapper instance by calling ::RoboCompLidar3DPub::newCallback_Lidar3DPub_pushLidarData.
 */
template<class T>
class CallbackNC_Lidar3DPub_pushLidarData : public Callback_Lidar3DPub_pushLidarData_Base, public ::IceInternal::OnewayCallbackNC<T>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception&);
    typedef void (T::*Sent)(bool);
    typedef void (T::*Response)();

    CallbackNC_Lidar3DPub_pushLidarData(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::OnewayCallbackNC<T>(obj, cb, excb, sentcb)
    {
    }
};

/**
 * Creates a callback wrapper instance that delegates to your object.
 * @param instance The callback object.
 * @param cb The success method of the callback object.
 * @param excb The exception method of the callback object.
 * @param sentcb The sent method of the callback object.
 * @return An object that can be passed to an asynchronous invocation of IceProxy::RoboCompLidar3DPub::Lidar3DPub::begin_pushLidarData.
 */
template<class T> Callback_Lidar3DPub_pushLidarDataPtr
newCallback_Lidar3DPub_pushLidarData(const IceUtil::Handle<T>& instance, void (T::*cb)(), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_Lidar3DPub_pushLidarData<T>(instance, cb, excb, sentcb);
}

/**
 * Creates a callback wrapper instance that delegates to your object.
 * @param instance The callback object.
 * @param excb The exception method of the callback object.
 * @param sentcb The sent method of the callback object.
 * @return An object that can be passed to an asynchronous invocation of IceProxy::RoboCompLidar3DPub::Lidar3DPub::begin_pushLidarData.
 */
template<class T> Callback_Lidar3DPub_pushLidarDataPtr
newCallback_Lidar3DPub_pushLidarData(const IceUtil::Handle<T>& instance, void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_Lidar3DPub_pushLidarData<T>(instance, 0, excb, sentcb);
}

/**
 * Creates a callback wrapper instance that delegates to your object.
 * @param instance The callback object.
 * @param cb The success method of the callback object.
 * @param excb The exception method of the callback object.
 * @param sentcb The sent method of the callback object.
 * @return An object that can be passed to an asynchronous invocation of IceProxy::RoboCompLidar3DPub::Lidar3DPub::begin_pushLidarData.
 */
template<class T> Callback_Lidar3DPub_pushLidarDataPtr
newCallback_Lidar3DPub_pushLidarData(T* instance, void (T::*cb)(), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_Lidar3DPub_pushLidarData<T>(instance, cb, excb, sentcb);
}

/**
 * Creates a callback wrapper instance that delegates to your object.
 * @param instance The callback object.
 * @param excb The exception method of the callback object.
 * @param sentcb The sent method of the callback object.
 * @return An object that can be passed to an asynchronous invocation of IceProxy::RoboCompLidar3DPub::Lidar3DPub::begin_pushLidarData.
 */
template<class T> Callback_Lidar3DPub_pushLidarDataPtr
newCallback_Lidar3DPub_pushLidarData(T* instance, void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_Lidar3DPub_pushLidarData<T>(instance, 0, excb, sentcb);
}

/**
 * Type-safe asynchronous callback wrapper class with cookie support used for calls to
 * IceProxy::RoboCompLidar3DPub::Lidar3DPub::begin_pushLidarData.
 * Create a wrapper instance by calling ::RoboCompLidar3DPub::newCallback_Lidar3DPub_pushLidarData.
 */
template<class T, typename CT>
class Callback_Lidar3DPub_pushLidarData : public Callback_Lidar3DPub_pushLidarData_Base, public ::IceInternal::OnewayCallback<T, CT>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception& , const CT&);
    typedef void (T::*Sent)(bool , const CT&);
    typedef void (T::*Response)(const CT&);

    Callback_Lidar3DPub_pushLidarData(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::OnewayCallback<T, CT>(obj, cb, excb, sentcb)
    {
    }
};

/**
 * Creates a callback wrapper instance that delegates to your object.
 * Use this overload when your callback methods receive a cookie value.
 * @param instance The callback object.
 * @param cb The success method of the callback object.
 * @param excb The exception method of the callback object.
 * @param sentcb The sent method of the callback object.
 * @return An object that can be passed to an asynchronous invocation of IceProxy::RoboCompLidar3DPub::Lidar3DPub::begin_pushLidarData.
 */
template<class T, typename CT> Callback_Lidar3DPub_pushLidarDataPtr
newCallback_Lidar3DPub_pushLidarData(const IceUtil::Handle<T>& instance, void (T::*cb)(const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_Lidar3DPub_pushLidarData<T, CT>(instance, cb, excb, sentcb);
}

/**
 * Creates a callback wrapper instance that delegates to your object.
 * Use this overload when your callback methods receive a cookie value.
 * @param instance The callback object.
 * @param excb The exception method of the callback object.
 * @param sentcb The sent method of the callback object.
 * @return An object that can be passed to an asynchronous invocation of IceProxy::RoboCompLidar3DPub::Lidar3DPub::begin_pushLidarData.
 */
template<class T, typename CT> Callback_Lidar3DPub_pushLidarDataPtr
newCallback_Lidar3DPub_pushLidarData(const IceUtil::Handle<T>& instance, void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_Lidar3DPub_pushLidarData<T, CT>(instance, 0, excb, sentcb);
}

/**
 * Creates a callback wrapper instance that delegates to your object.
 * Use this overload when your callback methods receive a cookie value.
 * @param instance The callback object.
 * @param cb The success method of the callback object.
 * @param excb The exception method of the callback object.
 * @param sentcb The sent method of the callback object.
 * @return An object that can be passed to an asynchronous invocation of IceProxy::RoboCompLidar3DPub::Lidar3DPub::begin_pushLidarData.
 */
template<class T, typename CT> Callback_Lidar3DPub_pushLidarDataPtr
newCallback_Lidar3DPub_pushLidarData(T* instance, void (T::*cb)(const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_Lidar3DPub_pushLidarData<T, CT>(instance, cb, excb, sentcb);
}

/**
 * Creates a callback wrapper instance that delegates to your object.
 * Use this overload when your callback methods receive a cookie value.
 * @param instance The callback object.
 * @param excb The exception method of the callback object.
 * @param sentcb The sent method of the callback object.
 * @return An object that can be passed to an asynchronous invocation of IceProxy::RoboCompLidar3DPub::Lidar3DPub::begin_pushLidarData.
 */
template<class T, typename CT> Callback_Lidar3DPub_pushLidarDataPtr
newCallback_Lidar3DPub_pushLidarData(T* instance, void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_Lidar3DPub_pushLidarData<T, CT>(instance, 0, excb, sentcb);
}

}

#endif

#include <IceUtil/PopDisableWarnings.h>
#endif
