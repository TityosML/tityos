#pragma once

#ifndef TITYOS_BUILD_SHARED
    #define TITYOS_EXPORT
    #define TITYOS_NO_EXPORT
#else
    #if defined(_WIN32) || defined(__CYGWIN__)
        #ifdef tityos_EXPORTS
            #define TITYOS_EXPORT __declspec(dllexport)
        #else
            #define TITYOS_EXPORT __declspec(dllimport)
        #endif
        #define TITYOS_NO_EXPORT
    #else
        #if __GNUC__ >= 4
            #define TITYOS_EXPORT __attribute__((visibility("default")))
            #define TITYOS_NO_EXPORT __attribute__((visibility("hidden")))
        #else
            #define TITYOS_EXPORT
            #define TITYOS_NO_EXPORT
        #endif
    #endif
#endif

#ifndef TITYOS_DEPRECATED
    #define TITYOS_DEPRECATED __attribute__((__deprecated__))
#endif

#ifndef TITYOS_DEPRECATED_EXPORT
    #define TITYOS_DEPRECATED_EXPORT TITYOS_EXPORT TITYOS_DEPRECATED
#endif

#ifndef TITYOS_DEPRECATED_NO_EXPORT
    #define TITYOS_DEPRECATED_NO_EXPORT TITYOS_NO_EXPORT TITYOS_DEPRECATED
#endif