/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <ctime>
#include <regex>
#include <sys/time.h>
#include <thread>
#include <vector>
#include "utils/host_log/log_time.h"

namespace npu::tile_fwk {
namespace {
std::string FormatReferenceTime(const time_t seconds)
{
    std::tm localTime{};
    if (localtime_r(&seconds, &localTime) == nullptr) {
        return {};
    }
    char buffer[32]{};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &localTime);
    return buffer;
}
std::string FormatReferenceFilenameTime(const time_t seconds)
{
    std::tm localTime{};
    if (localtime_r(&seconds, &localTime) == nullptr) {
        return {};
    }
    char buffer[32]{};
    std::strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", &localTime);
    return buffer;
}
void CheckConcurrentTimestamps(std::atomic<bool>* valid)
{
    for (int j = 0; j < 1000; ++j) {
        const std::string normal = GetCurrentTime();
        const std::string compact = GetCurrentTimeStr();
        if (normal.size() != 23 || compact.size() != 17 ||
            compact.find_first_not_of("0123456789") != std::string::npos) {
            *valid = false;
        }
    }
}
} // namespace

TEST(LogTimeTest, PreservesLocalTimestampFormat)
{
    timeval before{};
    ASSERT_EQ(gettimeofday(&before, nullptr), 0);
    const std::string actual = GetCurrentTime();
    timeval after{};
    ASSERT_EQ(gettimeofday(&after, nullptr), 0);
    ASSERT_TRUE(std::regex_match(actual, std::regex(R"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")));
    const std::string date = actual.substr(0, 19);
    EXPECT_GE(date, FormatReferenceTime(before.tv_sec));
    EXPECT_LE(date, FormatReferenceTime(after.tv_sec));
}

TEST(LogTimeTest, PreservesFilenameTimestampFormat)
{
    timeval before{};
    ASSERT_EQ(gettimeofday(&before, nullptr), 0);
    const std::string actual = GetCurrentTimeStr();
    timeval after{};
    ASSERT_EQ(gettimeofday(&after, nullptr), 0);
    ASSERT_TRUE(std::regex_match(actual, std::regex(R"(\d{17})")));
    const std::string date = actual.substr(0, 14);
    EXPECT_GE(date, FormatReferenceFilenameTime(before.tv_sec));
    EXPECT_LE(date, FormatReferenceFilenameTime(after.tv_sec));
}

TEST(LogTimeTest, ConcurrentTimestampGeneration)
{
    std::atomic<bool> valid{true};
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back(CheckConcurrentTimestamps, &valid);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_TRUE(valid.load());
}
namespace {
time_t UtcYearStart(const int year)
{
    std::tm date{};
    date.tm_year = year - 1900;
    date.tm_mon = 0;
    date.tm_mday = 1;
    return timegm(&date);
}

void ExpectCalendarMatchesLibc(const time_t seconds, const time_t timeZone, const int32_t dst)
{
    // gmtime_r is only the test oracle. Use an explicit offset rather than the
    // host timezone or historical DST rules; production caches these at startup.
    const time_t adjusted = seconds - timeZone + 3600 * dst;
    std::tm expected{};
    ASSERT_NE(gmtime_r(&adjusted, &expected), nullptr);
    std::tm actual{};
    detail::CalLocalTime(&actual, seconds, timeZone, dst);
    SCOPED_TRACE(::testing::Message() << "seconds=" << seconds << " timezone=" << timeZone << " dst=" << dst);
    EXPECT_EQ(actual.tm_year, expected.tm_year + 1900);
    EXPECT_EQ(actual.tm_mon, expected.tm_mon + 1);
    EXPECT_EQ(actual.tm_mday, expected.tm_mday);
    EXPECT_EQ(actual.tm_hour, expected.tm_hour);
    EXPECT_EQ(actual.tm_min, expected.tm_min);
    EXPECT_EQ(actual.tm_sec, expected.tm_sec);
    EXPECT_EQ(actual.tm_wday, expected.tm_wday);
    EXPECT_EQ(actual.tm_yday, expected.tm_yday);
    EXPECT_EQ(actual.tm_isdst, dst);
}
} // namespace

TEST(LogTimeTest, MatchesLibcEveryDayFrom1976Through2076)
{
    // Fixed coverage: 50 years before/after 2026, including both endpoint years.
    const time_t begin = UtcYearStart(1976);
    const time_t end = UtcYearStart(2077);
    ASSERT_NE(begin, static_cast<time_t>(-1));
    ASSERT_NE(end, static_cast<time_t>(-1));
    // timezone is seconds west of UTC; include whole, half and quarter hours.
    const time_t zones[] = {0, -8 * 3600, 5 * 3600, -19800, -20700, 12600, -14 * 3600, 12 * 3600};
    const time_t times[] = {0, 1, 12 * 3600 + 34 * 60 + 56, 86399};
    for (time_t day = begin; day < end; day += 86400) {
        for (const time_t zone : zones) {
            for (const time_t time : times) {
                ExpectCalendarMatchesLibc(day + time, zone, 0);
            }
        }
        // Exercise the implementation's fixed one-hour DST adjustment too.
        ExpectCalendarMatchesLibc(day, 5 * 3600, 1);
        ExpectCalendarMatchesLibc(day + 86399, -8 * 3600, 1);
    }
}

TEST(LogTimeTest, MatchesLibcAcrossCenturyLeapYearBoundaries)
{
    // 2000 is a leap year; 2100 is not. Check both sides of each day boundary
    // throughout February and March, beyond the main range for the latter.
    for (const int year : {2000, 2100}) {
        const time_t begin = UtcYearStart(year) + 31 * 86400;
        ASSERT_NE(UtcYearStart(year), static_cast<time_t>(-1));
        for (time_t day = begin; day < begin + 61 * 86400; day += 86400) {
            ExpectCalendarMatchesLibc(day - 1, 0, 0);
            ExpectCalendarMatchesLibc(day, 0, 0);
            ExpectCalendarMatchesLibc(day + 1, 0, 0);
        }
    }
}
} // namespace npu::tile_fwk
