#include "tools.hpp"
#include <ros/ros.h>
#include <Eigen/Eigenvalues>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/PoseArray.h>
#include <random>
#include <ctime>
#include <tf/transform_broadcaster.h>
#include "bavoxel.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <malloc.h>

using namespace std;

void pub_pl_func(pcl::PointCloud<PointType> &pl, ros::Publisher &pub)
{
  if (pl.empty())
  {
    ROS_WARN("PointCloud is empty, skipping publish.");
    return;
  }

  if (!pub)
  {
    ROS_ERROR("Publisher is not valid!");
    return;
  }
  pl.height = 1;
  pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "camera_init";
  output.header.stamp = ros::Time::now();
  pub.publish(output);
}

ros::Publisher pub_path, pub_show, pub_cute;

int read_pose(vector<double> &tims, PLM(3) & rots, PLV(3) & poss, string folder_name)
{
  string posename = folder_name + "Log/update_pose.csv";

  cout << posename << endl;
  ifstream inFile(posename);

  if (!inFile.is_open())
  {
    printf("open fail\n");
    return 0;
  }

  int pose_size = 0;
  string lineStr, str;
  Eigen::Matrix4d aff;
  vector<double> nums;

  int ord = 0;
  while (getline(inFile, lineStr))
  {
    ord++;
    stringstream ss(lineStr);
    while (getline(ss, str, ','))
      nums.push_back(stod(str));

    if (ord == 4)
    {
      for (int j = 0; j < 16; j++)
        aff(j) = nums[j];

      Eigen::Matrix4d affT = aff.transpose();

      rots.push_back(affT.block<3, 3>(0, 0));
      poss.push_back(affT.block<3, 1>(0, 3));
      tims.push_back(affT(3, 3));
      nums.clear();
      ord = 0;
      pose_size++;
    }
  }

  return pose_size;
}

void read_file(vector<IMUST> &x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls_, string &file_path_)
{
  PLV(3)
  poss;
  PLM(3)
  rots;
  vector<double> tims;
  int pose_size = read_pose(tims, rots, poss, file_path_);

  for (int m = 0; m < pose_size; m++)
  {
    ostringstream oss;
    oss << file_path_ << "PCDs/" << setw(6) << setfill('0') << m << ".pcd";
    string filename = oss.str();

    pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<pcl::PointXYZI> pl_tem;
    pcl::io::loadPCDFile(filename, pl_tem);
    double ratio = (double)(m + 1) / (double)pose_size * 100;
    ROS_INFO("[Debug] PCD Files Loaded: %.2f%% (%d/%d)", ratio, m + 1, pose_size);
    for (pcl::PointXYZI &pp : pl_tem.points)
    {
      PointType ap;
      ap.x = pp.x;
      ap.y = pp.y;
      ap.z = pp.z;
      ap.intensity = pp.intensity;
      pl_ptr->push_back(ap);
    }

    pl_fulls_.push_back(pl_ptr);

    IMUST curr;
    curr.R = rots[m];
    curr.p = poss[m];
    curr.t = tims[m];
    x_buf.push_back(curr);
  }
}

void saveMergedPCD(const std::vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls_, const std::string &output_path)
{
  pcl::PointCloud<PointType>::Ptr merged(new pcl::PointCloud<PointType>());

  for (const auto &cloud : pl_fulls_)
  {
    *merged += *cloud; // 点云叠加（也可以用 pcl::concatenatePointCloud）
  }

  pcl::io::savePCDFileBinary(output_path, *merged);
  std::cout << "合并后保存到 " << output_path << "，总点数：" << merged->size() << std::endl;
}

void saveSinglePointCloud(const pcl::PointCloud<PointType>& cloud, const std::string& output_path) {
  if (cloud.empty()) {
      std::cerr << "⚠️ 点云为空，未保存：" << output_path << std::endl;
      return;
  }

  pcl::io::savePCDFileBinary(output_path, cloud);
  std::cout << "点云保存成功：" << output_path << "（点数：" << cloud.size() << "）" << std::endl;
}

void data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls_)
{
  IMUST es0 = x_buf[0];
  for (uint i = 0; i < x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  pcl::PointCloud<PointType> pl_send, pl_path;
  int winsize = x_buf.size();
  for (int i = 0; i < winsize; i++)
  {
    pcl::PointCloud<PointType> pl_tem = *pl_fulls_[i];
    down_sampling_voxel(pl_tem, 0.05);
    pl_transform(pl_tem, x_buf[i]);
    pl_send += pl_tem;

    if ((i % 200 == 0 && i != 0) || i == winsize - 1)
    {
      pub_pl_func(pl_send, pub_show);
      pl_send.clear();
      sleep(0.5);
    }

    PointType ap;
    ap.x = x_buf[i].p.x();
    ap.y = x_buf[i].p.y();
    ap.z = x_buf[i].p.z();
    ap.curvature = i;
    pl_path.push_back(ap);
  }

  pub_pl_func(pl_path, pub_path);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "benchmark2");
  ros::NodeHandle n;
  pub_path = n.advertise<sensor_msgs::PointCloud2>("/map_path", 100);
  pub_show = n.advertise<sensor_msgs::PointCloud2>("/map_show", 100);
  pub_cute = n.advertise<sensor_msgs::PointCloud2>("/map_cute", 100);

  string prename, ofname;
  vector<IMUST> x_buf;
  vector<pcl::PointCloud<PointType>::Ptr> pl_fulls;

  n.param<double>("voxel_size", voxel_size, 1);
  string file_path;
  n.param<string>("file_path", file_path, "/home/ian/catkin_ws/src/BALM/datas/realworld/");

  read_file(x_buf, pl_fulls, file_path);

  IMUST es0 = x_buf[0];
  for (uint i = 0; i < x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  win_size = x_buf.size();
  printf("The size of poses: %d\n", win_size);

  data_show(x_buf, pl_fulls);
  // printf("Check the point cloud with the initial poses.\n");
  // printf("If no problem, input '1' to continue or '0' to exit...\n");
  // int a;
  // cin >> a;
  // if (a == 0)
  //   exit(0);

  pcl::PointCloud<PointType> pl_full, pl_surf, pl_path, pl_send;
  unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

  for (int i = 0; i < win_size; i++)
    cut_voxel(surf_map, *pl_fulls[i], x_buf[i], i);

  VOX_HESS voxhess;
  int total = surf_map.size(); // 获取总的元素数量
  int count = 0;               // 计数器

  for (auto iter = surf_map.begin(); iter != surf_map.end() && n.ok(); iter++)
  {
    iter->second->recut(win_size);
    iter->second->tras_opt(voxhess, win_size);
    iter->second->tras_display(pl_send, win_size);

    // 更新计数器并打印进度
    count++;
    double progress = (static_cast<double>(count) / total) * 100; // 计算进度百分比
    // ROS_INFO("Processing: %.2f%% (%d/%d)", progress, count, total); // 打印进度
  }

  pub_pl_func(pl_send, pub_cute);
  printf("\nThe planes (point association) cut by adaptive voxelization.\n");
  printf("If the planes are too few, the optimization will be degenerated and fail.\n");
  // printf("If no problem, input '1' to continue or '0' to exit...\n");
  // cin >> a;
  // if (a == 0)
  //   exit(0);
  // pl_send.clear();
  pub_pl_func(pl_send, pub_cute);

  if (voxhess.plvec_voxels.size() < 3 * x_buf.size())
  {
    printf("voxhess.plvec_voxels.size() = %ld\n", voxhess.plvec_voxels.size());
    printf("x_buf.size() = %ld\n", x_buf.size());
    printf("Initial error too large.\n");
    printf("TOO TOO LARGE!!!\n");
    printf("Please loose plane determination criteria for more planes.\n");
    printf("The optimization is terminated.\n");
    exit(0);
  }

  BALM2 opt_lsv;
  opt_lsv.damping_iter(x_buf, voxhess);

  for (auto iter = surf_map.begin(); iter != surf_map.end();)
  {
    delete iter->second;
    surf_map.erase(iter++);
  }
  surf_map.clear();

  printf("\nRefined point cloud is publishing...\n");
  malloc_trim(0);
  data_show(x_buf, pl_fulls);
  printf("Refined point cloud is published.\n");
  saveMergedPCD(pl_fulls, file_path+"output/merged.pcd");
  saveSinglePointCloud(pl_send, file_path+"output/merged_single.pcd");
  pl_send.clear();
  saveSinglePointCloud(pl_path, file_path+"output/path.pcd");
  // ros::spin();
  return 0;
}
