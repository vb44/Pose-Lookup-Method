#include "utils.hpp"

namespace utils {

    Eigen::Matrix4d homogeneous(double roll, double pitch, double yaw, 
                                double x, double y, double z)
    {
        Eigen::Matrix4d T;
        T.setZero();

        T(0,0) = cos(yaw) * cos(pitch);
        T(0,1) = cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll);
        T(0,2) = cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll);
        T(0,3) = x;
        T(1,0) = sin(yaw) * cos(pitch);
        T(1,1) = sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll);
        T(1,2) = sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll);
        T(1,3) = y;
        T(2,0) = -sin(pitch);
        T(2,1) = cos(pitch) * sin(roll);
        T(2,2) = cos(pitch) * cos(roll);
        T(2,3) = z;
        T(3,3) = 1;

        return T;
    }

    std::vector<double> hom2rpyxyz(const Eigen::Matrix4d &T)
    {
        double roll = atan2(T(2,1), T(2,2));
        double pitch = asin(-T(2,0));
        double yaw = atan2(T(1,0), T(0,0));
        double x = T(0,3);
        double y = T(1,3);
        double z = T(2,3);
        std::vector<double> result = {roll, pitch, yaw, x, y, z};
        return result;
    }

    bool compareStrings(const std::string &a, const std::string &b)
    {
        std::string delimiterStart = "/";
        std::string delimiterEnd = ".bin";
        
        std::string aNum = a.substr(a.find_last_of(delimiterStart) +
                                                delimiterStart.size(),
                                                a.size());
        std::string bNum = b.substr(b.find_last_of(delimiterStart) +
                                                delimiterStart.size(),
                                                b.size());
        aNum = aNum.substr(0, aNum.find(delimiterEnd));
        bNum = bNum.substr(0, bNum.find(delimiterEnd));

        return stol(aNum) < stol(bNum);
    }

    void printProgress(double percentage) {
        // code from https://stackoverflow.com/questions/14539867/
        //           how-to-display-a-progress-indicator-in-pure-c-c-cout-printf
        int progressBarWidth = 60;
        char progressBarString[] = 
            "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||";
        int val = (int) (percentage * 100);
        int lpad = (int) (percentage * progressBarWidth);
        int rpad = progressBarWidth - lpad;
        printf("\r%3d%% [%.*s%*s]", val, lpad, progressBarString, rpad, "");
        fflush(stdout);
    }
}