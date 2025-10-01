#ifndef INCLUDE_TYE_SP_INCLUDES_H
#define INCLUDE_TYE_SP_INCLUDES_H

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include <iostream>
#include <exception>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <limits>
#include <chrono>
#include <vector>
#include <tuple>
#include <filesystem>
#include <thread>
#include <mutex>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <csignal>
#include <ctime>

#include <unistd.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <GL/gl.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <bsoncxx/json.hpp>
#include <mongocxx/exception/exception.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/uri.hpp>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#ifdef TYE_STREAM_PROCESSOR
#include "sm_api.h"
#endif

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_TYE_SP_INCLUDES_H
