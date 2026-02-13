#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef tityos_EXPORTS
        #define TITYOS_API __declspec(dllexport)
    #else
        #define TITYOS_API __declspec(dllimport)
    #endif
#else
    #if __GNUC__ >= 4
        #define TITYOS_API __attribute__((visibility("default")))
    #else
        #define TITYOS_API
    #endif
#endif

#ifndef TITYOS_DEPRECATED
    #define TITYOS_DEPRECATED __attribute__((__deprecated__))
#endif