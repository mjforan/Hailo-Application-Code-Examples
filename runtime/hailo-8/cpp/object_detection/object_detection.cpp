/**
 * Copyright (c) 2020-2025 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file async_infer_basic_example.cpp
 * This example demonstrates the Async Infer API usage with a specific model.
 **/
#include "toolbox.hpp"
using namespace hailo_utils;

#include "hailo_infer.hpp"
#include "utils.hpp"

#include "ByteTrack/BYTETracker.h"
#include "ByteTrack/Object.h"
#include "ByteTrack/Rect.h"

namespace fs = std::filesystem;

/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue =
    std::make_shared<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue =
    std::make_shared<BoundedTSQueue<InferenceResult>>(MAX_QUEUE_SIZE);

// Task-specific preprocessing callback
void preprocess_callback(const std::vector<cv::Mat>& org_frames, 
                                        std::vector<cv::Mat>& preprocessed_frames, 
                                        uint32_t target_width, uint32_t target_height) {
    for (const auto& frame : org_frames) {
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(target_width, target_height));
        preprocessed_frames.push_back(resized_frame);
    }
}

// Task-specific postprocessing callback
void postprocess_callback(const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>& output_data_and_infos,
                                        std::shared_ptr<rclcpp::Node> ros_node,
                                        std::shared_ptr<byte_track::BYTETracker> tracker,
                                        int input_w,
                                        int input_h
) {
    auto ros_publisher = ros_node->create_publisher<vision_msgs::msg::Detection2DArray>("/hailo/detections", 10);
    size_t class_count = 80; // 80 classes in COCO dataset
    auto bboxes = parse_nms_data(output_data_and_infos[0].first, class_count);
    //draw_bounding_boxes(frame_to_draw, bboxes);

    // Convert format for bytetrack
    std::vector<byte_track::Object> objects;
    for (const NamedBbox & detection : bboxes) {
        objects.push_back(
            byte_track::Object(byte_track::Rect(
                input_w*(detection.bbox.x_min + detection.bbox.x_max)/2,
                input_h*(detection.bbox.y_min + detection.bbox.y_max)/2,
                input_w*(detection.bbox.x_max - detection.bbox.x_min),
                input_h*(detection.bbox.y_max - detection.bbox.y_min)
            ),
            detection.class_id,
            detection.bbox.score)
        );
    }

    // Update tracker
    const std::vector<byte_track::BYTETracker::STrackPtr> outputs = tracker->update(objects);

    // Construct ROS message and publish
    vision_msgs::msg::Detection2DArray det_arr_msg;
    // TODO get timestamp of original image - not currently possible with Hailo tools
    det_arr_msg.header.stamp = ros_node->get_clock()->now();
    det_arr_msg.header.frame_id = "hailo_frame";
    for (const auto &tracked_detection : outputs)
    {
        vision_msgs::msg::Detection2D det_msg;
        det_msg.header.stamp = det_arr_msg.header.stamp;
        det_msg.header.frame_id = det_arr_msg.header.frame_id;
        det_msg.id = tracked_detection->getTrackId();
        const auto &rect = tracked_detection->getRect();
        det_msg.bbox.center.position.x = rect.x();
        det_msg.bbox.center.position.y = rect.y();
        det_msg.bbox.size_x            = rect.width();
        det_msg.bbox.size_y            = rect.height();
        det_msg.results.push_back(vision_msgs::msg::ObjectHypothesisWithPose());
        det_msg.results[0].hypothesis.class_id = "0";//std::to_string(detection.class_id); // TODO
        det_msg.results[0].hypothesis.score    = tracked_detection->getScore();
        det_arr_msg.detections.push_back(det_msg);
    }

    ros_publisher->publish(det_arr_msg);
}

int main(int argc, char** argv)
{
    CommandLineArgs args;

    rclcpp::init(argc, argv);
    auto ros_node = std::make_shared<rclcpp::Node>("hailo_detect");
    ros_node->declare_parameter("model", "/model.hef");
    args.detection_hef = ros_node->get_parameter("model").as_string();
    ros_node->declare_parameter("source", "/dev/video0");
    args.input_path = ros_node->get_parameter("source").as_string();
    ros_node->declare_parameter("batch_size", 1);
    size_t batch_size = ros_node->get_parameter("batch_size").as_int();

    std::chrono::duration<double> inference_time;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();
    double org_height, org_width;
    cv::VideoCapture capture;
    size_t frame_count;
    InputType input_type;

    HailoInfer model(args.detection_hef, batch_size);
    input_type = determine_input_type(args.input_path, std::ref(capture), org_height, org_width, frame_count, batch_size);

    capture.open(args.input_path, cv::CAP_ANY);
    if (!capture.isOpened()) {
        throw std::runtime_error("Unable to read input file");
    }
    double fps = capture.get(cv::CAP_PROP_FPS);

    // buffer missing track for 1s
    auto tracker = std::make_shared<byte_track::BYTETracker>(fps, 1.0*fps);

    auto preprocess_thread = std::async(run_preprocess,
                                        args.input_path,
                                        args.detection_hef,
                                        std::ref(model),
                                        std::ref(input_type),
                                        std::ref(capture),
                                        batch_size,
                                        preprocessed_batch_queue,
                                        preprocess_callback);

    auto inference_thread = std::async(run_inference_async,
                                    std::ref(model),
                                    std::ref(inference_time),
                                    preprocessed_batch_queue,
                                    results_queue);

    auto output_parser_thread = std::async(run_post_process,
                                std::ref(input_type),
                                org_height,
                                org_width,
                                frame_count,
                                std::ref(capture),
                                results_queue,
                                ros_node,
                                tracker,
                                postprocess_callback);

    hailo_status status = wait_and_check_threads(
        preprocess_thread,    "Preprocess",
        inference_thread,     "Inference",
        output_parser_thread, "Postprocess "
    );
    if (HAILO_SUCCESS != status) {
        return status;
    }

    if(!input_type.is_camera) {
        std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
        print_inference_statistics(inference_time, frame_count, t_end - t_start);
    }

    rclcpp::shutdown();
    return HAILO_SUCCESS;
}
