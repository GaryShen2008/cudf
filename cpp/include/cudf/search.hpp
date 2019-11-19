/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

#include <vector>

namespace cudf {
namespace experimental {

/**---------------------------------------------------------------------------*
 * @brief Find smallest indices in a sorted table where values should be 
 *  inserted to maintain order
 * 
 * For each row v in @p values, find the first index in @p t where
 *  inserting the row will maintain the sort order of @p t
 * 
 * Example:
 * 
 *  Single column:
 *      idx      0   1   2   3   4
 *   column = { 10, 20, 20, 30, 50 }
 *   values = { 20 }
 *   result = {  1 }
 * 
 *  Multi Column:
 *      idx        0    1    2    3    4
 *   t      = {{  10,  20,  20,  20,  20 },
 *             { 5.0,  .5,  .5,  .7,  .7 },
 *             {  90,  77,  78,  61,  61 }}
 *   values = {{ 20 },
 *             { .7 },
 *             { 61 }}
 *   result =  {  3 }
 * 
 * @param t               Table to search
 * @param values          Find insert locations for these values
 * @param column_order    Vector of column sort order
 * @param null_precedence Vector of null_precedence enums
 * values
 * @param mr              Device memory resource to use for device memory allocation
 * @return std::unique_ptr<column> A non-nullable column of cudf::size_type elements
 * containing the insertion points.
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> lower_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Find largest indices in a sorted table where values should be 
 *  inserted to maintain order
 * 
 * For each row v in @p values, find the last index in @p t where
 *  inserting the row will maintain the sort order of @p t
 * 
 * Example:
 * 
 *  Single Column:
 *      idx      0   1   2   3   4
 *   column = { 10, 20, 20, 30, 50 }
 *   values = { 20 }
 *   result = {  3 }
 * 
 *  Multi Column:
 *    idx        0    1    2    3    4
 *   t      = {{  10,  20,  20,  20,  20 },
 *             { 5.0,  .5,  .5,  .7,  .7 },
 *             {  90,  77,  78,  61,  61 }}
 *   values = {{ 20 },
 *             { .7 },
 *             { 61 }}
 *   result =  {  5  *   * 
 * @param column          Table to search
 * @param values          Find insert locations for these values
 * @param column_order    Vector of column sort order
 * @param null_precedence Vector of null_precedence enums
 * values
 * @param mr              Device memory resource to use for device memory allocation
 * @return std::unique_ptr<column> A non-nullable column of cudf::size_type elements
 * containing the insertion points.
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> upper_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Find if the `value` is present in the `col`
 *
 * @throws cudf::logic_error
 * If `col.type() != values.type()`
 *
 * @example:
 *
 *  Single Column:
 *      idx      0   1   2   3   4
 *      col = { 10, 20, 20, 30, 50 }
 *  Scalar:
 *   value = { 20 }
 *   result = true
 *
 * @param col      A column object
 * @param value    A scalar value to search for in `col`
 * @param mr       Device memory resource to use for device memory allocation
 *
 * @return bool    If `value` is found in `column` true, else false.
 *---------------------------------------------------------------------------**/
bool contains(column_view const& col, scalar const& value,
              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace experimental
} // namespace cudf

