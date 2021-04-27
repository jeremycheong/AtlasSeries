#pragma once

#include <iostream>
#include <experimental/filesystem>

#include <string>
#include <vector>
#include <sstream>

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "\033[33m[ERROR]  [%s:%d] %s: " fmt "\n\033[0m", __FILE__, __LINE__, __FUNCTION__, ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "\033[31m[ERROR]  [%s:%d] %s: " fmt "\n\033[0m", __FILE__, __LINE__, __FUNCTION__, ##args)

namespace fs = std::experimental::filesystem;

class Path
{
public:
    Path(const std::string &path_str);
    Path(const char* path_char);
    ~Path()=default;
    static bool isexist(const std::string &path_str);
    static std::vector<std::string> split(const std::string &path_str, const std::string &split_word="/");
    static std::vector<std::string> splitext(const std::string &path_str);
    static uint32_t getsize(const std::string &file_path);
    static bool isfile(const std::string &path_str);
    static bool isdir(const std::string &path_str);
    static Path join(const std::string &path1_str, const std::string &path2_str);

    static std::vector<std::string> getfiles(const std::string &path_str, const std::string &text_word);
    static std::string parentpath(const std::string &path_str);

    inline bool isexist() const
    {
        return fs::exists(path_);
    }

    inline uint32_t getsize() const
    {
        if (!isexist() || !isfile(path_))
        {
            ERROR_LOG("The file: %s is not exist or not a file", path_.c_str());
            return 0;
        }
        return fs::file_size(path_);
    }

    inline std::string tostring() const 
    {
        return path_;
    }

    Path operator / (const Path &path_obj) const;
    Path operator / (const std::string &path_str) const;
    Path operator / (const char* path_char) const;


    void operator /= (const Path &path_obj);
    void operator /= (const std::string &path_str);
    void operator /= (const char* path_char);


private:
    Path(/* args */){};
    
    std::string path_;
};

Path::Path(const std::string &path_str)
    : path_(path_str)
{

}

Path::Path(const char* path_char)
    : path_(std::move(std::string(path_char)))
{

}

bool Path::isexist(const std::string &path_str)
{
    return fs::exists(path_str);
}


std::vector<std::string> Path::split(const std::string &path_str, const std::string &split_word)
{
    std::vector<std::string> result;
    std::istringstream iss(path_str);
    std::string words;
    for (std::string w; iss >> w; )
    {
        if (w != split_word)
        {
            words.append(w);
            continue;
        }
        result.push_back(words);
        words.clear();
    }
        
    return result;
}

std::vector<std::string> Path::splitext(const std::string &path_str)
{
    std::vector<std::string> results;
    fs::path file_path(path_str);
    if (!fs::exists(file_path))
    {
        ERROR_LOG("file path: %s is not exist!", file_path.c_str());
        return results;
    }

    if (fs::is_directory(file_path))
    {
        ERROR_LOG("file path %s is not a file", file_path.string().c_str());
        return results;
    }

    // results.assign({(file_path.parent_path()/file_path.stem()).string(), file_path.extension().string()});
    results.assign({file_path.stem().string(), file_path.extension().string()});
    
    return results;    
}

uint32_t Path::getsize(const std::string &file_path)
{
    if (isfile(file_path))
        return fs::file_size(file_path);
    else
    {
        ERROR_LOG("file_path: %s is not a file", file_path.c_str());
        return 0;
    }
}

bool Path::isfile(const std::string &path_str)
{
    if (!fs::exists(path_str))
    {
        ERROR_LOG("path: %s is not exist", path_str.c_str());
        return false;
    }
    
    return fs::is_directory(path_str) ? false : true;
}

bool Path::isdir(const std::string &path_str)
{
    if (!fs::exists(path_str))
    {
        ERROR_LOG("path: %s is not exist", path_str.c_str());
        return false;
    }
    
    return fs::is_directory(path_str) ? true : false;

}

Path Path::join(const std::string &path1_str, const std::string &path2_str)
{
    std::string new_path;
    new_path.append(path1_str).append("/").append(path2_str);

    return Path(new_path);
}

Path Path::operator / (const Path &path_obj) const
{
    return join(this->path_, path_obj.tostring());
}

Path Path::operator / (const std::string &path_str) const
{
    return join(this->path_, path_str);
}

Path Path::operator / (const char* path_char) const
{
    return join(this->path_, path_char);
}


void Path::operator /= (const Path &path_obj)
{
    this->path_.append("/").append(path_obj.tostring());
}

void Path::operator /= (const std::string &path_str)
{
    this->path_.append("/").append(path_str);
}

void Path::operator /= (const char* path_char)
{

}

std::vector<std::string> Path::getfiles(const std::string &path_str, const std::string &text_word)
{
    std::vector<std::string> results;
    fs::path dir_path(path_str);
    if (!fs::is_directory(dir_path))
    {
        results = {path_str};
        return results;
    }

    for (auto i = fs::directory_iterator(dir_path); i != fs::directory_iterator(); i ++)
    {
        if (!fs::is_directory(i->path()))
        {
            std::string file_name = i->path().filename().string();
            if (text_word.empty())
                results.emplace_back(file_name);
            else
            {
                std::string file_path = path_str + "/" + file_name;
                if (splitext(file_path)[1] == text_word)
                    results.emplace_back(file_name);
            }
        }
    }

    return results;
}

std::string Path::parentpath(const std::string &path_str)
{
    if (!isfile(path_str))
    {
        return path_str;
    }

    fs::path file_path(path_str);
    return file_path.parent_path().string();
}
