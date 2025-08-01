import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

st.set_page_config(page_title="Financial Statement Accuracy Checker", layout="wide")
st.title("📊 Financial Statement Accuracy Checker")

# Sidebar option
comparison_mode = st.sidebar.radio("Select Comparison Mode", ["Ground Truth vs Extracted", "Pairwise Result Comparison (R1 vs R2)"])

# Shared helper functions
def normalize_field_name(field):
    """Normalize field names for better matching"""
    return str(field).strip().lower().replace("_", " ").replace("-", " ")

def normalize(field):
    """Original normalize function for ground truth comparison"""
    return str(field).strip().lower()

def to_float(val):
    """Convert value to float, handling various formats"""
    if pd.isna(val) or val == "" or val is None:
        return None
    try:
        val_str = str(val).replace(",", "").replace("(", "-").replace(")", "").replace("$", "").strip()
        return float(val_str) if val_str != "" else None
    except:
        return None

def calculate_percentage_diff(val1, val2):
    """Calculate percentage difference between two values"""
    if pd.isna(val1) or pd.isna(val2) or val1 is None or val2 is None:
        return None
    if val2 == 0:
        return np.inf if val1 != 0 else 0
    return ((val1 - val2) / abs(val2)) * 100

def generate_summary_stats(df):
    """Generate summary statistics for a dataframe"""
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        stats[col] = {
            'count': df[col].count(),
            'null_count': df[col].isnull().sum(),
            'completeness_rate': (df[col].count() / len(df)) * 100,
            'mean': df[col].mean() if df[col].count() > 0 else None,
            'std': df[col].std() if df[col].count() > 0 else None,
            'min': df[col].min() if df[col].count() > 0 else None,
            'max': df[col].max() if df[col].count() > 0 else None
        }
    
    return stats

if comparison_mode == "Ground Truth vs Extracted":
    st.subheader("🎯 Ground Truth vs Extracted Results Analysis")
    
    # Upload files
    col1, col2 = st.columns(2)
    with col1:
        gt_file = st.file_uploader("📥 Upload Ground Truth CSV", type="csv", key="gt_file")
        if gt_file:
            st.success(f"✅ Ground Truth uploaded: {gt_file.name}")
    
    with col2:
        extract_file = st.file_uploader("📥 Upload Extracted File", type=["xlsx", "csv"], key="extract_file")
        if extract_file:
            st.success(f"✅ Extracted file uploaded: {extract_file.name}")

    if gt_file and extract_file:
        try:
            # Load data with enhanced error handling
            gt_df = pd.read_csv(gt_file)
            
            # Validate ground truth structure
            required_gt_cols = ['Category', 'Field']
            missing_gt_cols = [col for col in required_gt_cols if col not in gt_df.columns]
            if missing_gt_cols:
                st.error(f"❌ Ground Truth file missing required columns: {missing_gt_cols}")
                st.stop()

            if extract_file.name.endswith('.csv'):
                df = pd.read_csv(extract_file)
            else:
                # Handle Excel files with multiple sheets
                xls = pd.ExcelFile(extract_file)
                if len(xls.sheet_names) > 1:
                    selected_sheet = st.selectbox("Select Excel sheet:", xls.sheet_names)
                    df = xls.parse(selected_sheet)
                else:
                    df = xls.parse(xls.sheet_names[0])
            
            # Validate extracted file structure
            if 'Field' not in df.columns:
                st.error("❌ Extracted file missing required 'Field' column")
                st.stop()

            # Display dataset overview
            st.markdown("---")
            st.subheader("📋 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Ground Truth Records", len(gt_df))
            with col2:
                st.metric("Extracted Records", len(df))
            with col3:
                year_cols_gt = [col for col in gt_df.columns if col not in ['Category', 'Field']]
                st.metric("GT Year Columns", len(year_cols_gt))
            with col4:
                year_cols_ext = [col for col in df.columns if col != 'Field']
                st.metric("Extracted Year Columns", len(year_cols_ext))

            # Enhanced data preparation
            gt_long = gt_df.melt(id_vars=['Category', 'Field'], var_name='Year', value_name='Value_GT')
            gt_long['Year'] = gt_long['Year'].astype(str).str.strip()
            gt_long['Field'] = gt_long['Field'].astype(str).str.strip()

            # Handle field normalization for extracted data
            df['Normalized_Field'] = df['Field'].astype(str).apply(
                lambda x: x.replace(" Current Year", "").replace(" Prior Year", "").strip()
            )
            
            # Create long format for extracted data
            df_long = df.melt(id_vars=['Field', 'Normalized_Field'], var_name='Year', value_name='Value_Extracted')
            df_long['Year'] = df_long['Year'].astype(str).str.strip()
            df_long['Normalized_Field'] = df_long['Normalized_Field'].astype(str).str.strip()

            # Apply normalization for matching
            gt_long['Field_Normalized'] = gt_long['Field'].apply(normalize)
            df_long['Field_Normalized'] = df_long['Normalized_Field'].apply(normalize)

            # Enhanced merging with better matching
            comparison_df = pd.merge(
                df_long,
                gt_long,
                left_on=['Field_Normalized', 'Year'],
                right_on=['Field_Normalized', 'Year'],
                how='outer',
                suffixes=('_ext', '_gt')
            )

            # Enhanced value comparison
            comparison_df['Value_Extracted'] = comparison_df['Value_Extracted'].apply(to_float)
            comparison_df['Value_GT'] = comparison_df['Value_GT'].apply(to_float)
            
            # More sophisticated matching logic
            def values_match(val1, val2, tolerance=0.01):
                if pd.isna(val1) and pd.isna(val2):
                    return True
                if pd.isna(val1) or pd.isna(val2):
                    return False
                return abs(val1 - val2) <= tolerance
            
            comparison_df['Exact_Match'] = comparison_df.apply(
                lambda row: values_match(row['Value_Extracted'], row['Value_GT']), axis=1
            )
            
            comparison_df['Tolerance_Match'] = comparison_df.apply(
                lambda row: values_match(row['Value_Extracted'], row['Value_GT'], tolerance=1.0), axis=1
            )

            # Calculate percentage differences
            comparison_df['Percentage_Diff'] = comparison_df.apply(
                lambda row: calculate_percentage_diff(row['Value_Extracted'], row['Value_GT']), axis=1
            )

            # Enhanced accuracy metrics
            st.markdown("---")
            st.subheader("🎯 Accuracy Metrics")
            
            # Filter for valid ground truth values
            valid_gt = comparison_df['Value_GT'].notna()
            total_fields = valid_gt.sum()
            
            if total_fields > 0:
                exact_matches = comparison_df.loc[valid_gt, 'Exact_Match'].sum()
                tolerance_matches = comparison_df.loc[valid_gt, 'Tolerance_Match'].sum()
                
                # Coverage metrics
                extracted_coverage = comparison_df.loc[valid_gt, 'Value_Extracted'].notna().sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    exact_accuracy = (exact_matches / total_fields) * 100
                    st.metric("🎯 Exact Accuracy", f"{exact_accuracy:.1f}%", f"{exact_matches}/{total_fields}")
                
                with col2:
                    tolerance_accuracy = (tolerance_matches / total_fields) * 100
                    st.metric("📊 Tolerance Accuracy (±1)", f"{tolerance_accuracy:.1f}%", f"{tolerance_matches}/{total_fields}")
                
                with col3:
                    coverage = (extracted_coverage / total_fields) * 100
                    st.metric("📈 Extraction Coverage", f"{coverage:.1f}%", f"{extracted_coverage}/{total_fields}")
                
                with col4:
                    # Calculate average absolute percentage error for matched values
                    valid_comparisons = comparison_df[
                        comparison_df['Value_GT'].notna() & 
                        comparison_df['Value_Extracted'].notna() &
                        comparison_df['Percentage_Diff'].notna() &
                        (comparison_df['Percentage_Diff'] != np.inf)
                    ]
                    if not valid_comparisons.empty:
                        mape = valid_comparisons['Percentage_Diff'].abs().mean()
                        st.metric("📉 Mean Abs % Error", f"{mape:.2f}%")
                    else:
                        st.metric("📉 Mean Abs % Error", "N/A")
                
                # Category-wise analysis
                if 'Category' in comparison_df.columns:
                    st.markdown("---")
                    st.subheader("📂 Category-wise Performance")
                    
                    category_stats = []
                    for category in comparison_df['Category'].dropna().unique():
                        cat_data = comparison_df[
                            (comparison_df['Category'] == category) & 
                            comparison_df['Value_GT'].notna()
                        ]
                        
                        if len(cat_data) > 0:
                            cat_total = len(cat_data)
                            cat_exact = cat_data['Exact_Match'].sum()
                            cat_tolerance = cat_data['Tolerance_Match'].sum()
                            cat_coverage = cat_data['Value_Extracted'].notna().sum()
                            
                            category_stats.append({
                                'Category': category,
                                'Total_Fields': cat_total,
                                'Exact_Matches': cat_exact,
                                'Exact_Accuracy_%': (cat_exact / cat_total) * 100,
                                'Tolerance_Matches': cat_tolerance,
                                'Tolerance_Accuracy_%': (cat_tolerance / cat_total) * 100,
                                'Coverage_%': (cat_coverage / cat_total) * 100
                            })
                    
                    if category_stats:
                        category_df = pd.DataFrame(category_stats)
                        st.dataframe(category_df, use_container_width=True)

                # Enhanced mismatch analysis
                st.markdown("---")
                st.subheader("🔍 Detailed Mismatch Analysis")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    mismatch_type = st.radio(
                        "Show mismatches:",
                        ["Exact mismatches only", "Outside tolerance (±1)", "All discrepancies"]
                    )
                
                with col2:
                    if 'Category' in comparison_df.columns:
                        categories = ['All'] + list(comparison_df['Category'].dropna().unique())
                        selected_category = st.selectbox("Filter by category:", categories)
                    else:
                        selected_category = 'All'
                
                # Apply filters
                if mismatch_type == "Exact mismatches only":
                    mismatches = comparison_df[~comparison_df['Exact_Match'] & comparison_df['Value_GT'].notna()]
                elif mismatch_type == "Outside tolerance (±1)":
                    mismatches = comparison_df[~comparison_df['Tolerance_Match'] & comparison_df['Value_GT'].notna()]
                else:
                    mismatches = comparison_df[comparison_df['Value_GT'].notna()]
                
                if selected_category != 'All':
                    mismatches = mismatches[mismatches['Category'] == selected_category]
                
                # Select relevant columns for display
                display_cols = ['Field_ext', 'Year', 'Value_Extracted', 'Value_GT', 'Percentage_Diff']
                if 'Category' in mismatches.columns:
                    display_cols.insert(0, 'Category')
                
                mismatches_display = mismatches[display_cols].copy()
                
                if len(mismatches_display) > 0:
                    # Sort by absolute percentage difference
                    if 'Percentage_Diff' in mismatches_display.columns:
                        mismatches_display['Abs_Percentage_Diff'] = mismatches_display['Percentage_Diff'].abs()
                        mismatches_display = mismatches_display.sort_values('Abs_Percentage_Diff', ascending=False, na_position='last')
                        mismatches_display = mismatches_display.drop('Abs_Percentage_Diff', axis=1)
                    
                    st.dataframe(mismatches_display, use_container_width=True)
                    
                    # Summary of mismatches
                    st.info(f"Found {len(mismatches_display)} mismatches based on selected criteria")
                else:
                    st.success("🎉 No mismatches found based on selected criteria!")

                # Enhanced export options
                st.markdown("---")
                st.subheader("📤 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV export
                    csv_data = comparison_df.to_csv(index=False).encode()
                    st.download_button(
                        "📄 Download Full Comparison (CSV)",
                        csv_data,
                        f"gt_vs_extracted_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                
                with col2:
                    # Excel export with multiple sheets
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Summary sheet
                        summary_data = {
                            'Metric': ['Total Fields', 'Exact Matches', 'Exact Accuracy %', 
                                      'Tolerance Matches', 'Tolerance Accuracy %', 'Coverage %', 'MAPE %'],
                            'Value': [total_fields, exact_matches, round(exact_accuracy, 2), 
                                     tolerance_matches, round(tolerance_accuracy, 2), round(coverage, 2),
                                     round(mape, 2) if 'mape' in locals() else 'N/A']
                        }
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Full comparison
                        comparison_df.to_excel(writer, sheet_name='Full_Comparison', index=False)
                        
                        # Category stats if available
                        if 'category_df' in locals():
                            category_df.to_excel(writer, sheet_name='Category_Analysis', index=False)
                        
                        # Mismatches
                        if len(mismatches_display) > 0:
                            mismatches_display.to_excel(writer, sheet_name='Mismatches', index=False)
                    
                    excel_data = output.getvalue()
                    st.download_button(
                        "📊 Download Detailed Report (Excel)",
                        excel_data,
                        f"gt_vs_extracted_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                st.success("✅ Analysis complete! Review the metrics above and download detailed reports.")
            
            else:
                st.warning("⚠️ No valid ground truth values found for comparison.")

        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.info("Please ensure your files have the correct format:")
            st.markdown("""
            **Ground Truth CSV should have:**
            - Category column
            - Field column  
            - Year columns (e.g., 2021, 2022, 2023)
            
            **Extracted file (Excel or CSV) should have:**
            - Field column
            - Year columns with data
            """)

    else:
        st.info("⬆️ Please upload both Ground Truth and Extracted files to proceed.")
        with st.expander("📖 Expected File Format"):
            st.markdown("""
            **Ground Truth CSV:**
            ```
            Category,Field,2021,2022,2023
            Revenue,Total Revenue,1000000,1100000,1200000
            Expenses,Operating Expenses,800000,850000,900000
            ```

            **Extracted Excel or CSV:**
            ```
            Field,2021,2022,2023
            Total Revenue Current Year,1000000,1050000,1200000
            Operating Expenses Prior Year,780000,850000,920000
            ```
            """)

elif comparison_mode == "Pairwise Result Comparison (R1 vs R2)":
    st.subheader("🔁 Comprehensive Analysis: Pairwise Result Comparison")
    
    # File upload
    col1, col2 = st.columns(2)
    with col1:
        r1_file = st.file_uploader("📂 Upload First Result", type=["csv", "xlsx"], key="r1")
        if r1_file:
            st.success(f"✅ First Result uploaded: {r1_file.name}")
    
    with col2:
        r2_file = st.file_uploader("📂 Upload Second Results", type=["csv", "xlsx"], key="r2")
        if r2_file:
            st.success(f"✅ R2 uploaded: {r2_file.name}")
    
    if r1_file and r2_file:
        try:
            # Load data
            if r1_file.name.endswith('.csv'):
                df_r1 = pd.read_csv(r1_file)
            else:
                df_r1 = pd.read_excel(r1_file)
                
            if r2_file.name.endswith('.csv'):
                df_r2 = pd.read_csv(r2_file)
            else:
                df_r2 = pd.read_excel(r2_file)
            
            # Display basic info
            st.markdown("---")
            st.subheader("📋 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R1 Rows", len(df_r1))
            with col2:
                st.metric("R1 Columns", len(df_r1.columns))
            with col3:
                st.metric("R2 Rows", len(df_r2))
            with col4:
                st.metric("R2 Columns", len(df_r2.columns))
            
            # === FIELD NAME ANALYSIS ===
            st.markdown("---")
            st.subheader("🏷️ Field Name Analysis")
            
            # Assuming 'Field' column exists
            if 'Field' in df_r1.columns and 'Field' in df_r2.columns:
                fields_r1 = set(df_r1['Field'].dropna().astype(str))
                fields_r2 = set(df_r2['Field'].dropna().astype(str))
                
                # Field comparison metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Fields in R1", len(fields_r1))
                with col2:
                    st.metric("Fields in R2", len(fields_r2))
                with col3:
                    st.metric("Common Fields", len(fields_r1.intersection(fields_r2)))
                with col4:
                    consistency_rate = len(fields_r1.intersection(fields_r2)) / len(fields_r1.union(fields_r2)) * 100
                    st.metric("Field Consistency", f"{consistency_rate:.1f}%")
                
                # Detailed field comparison
                tab1, tab2, tab3 = st.tabs(["🔄 Field Mapping", "➕ Unique to R1", "➕ Unique to R2"])
                
                with tab1:
                    st.write("**Common Fields Between R1 and R2:**")
                    common_fields = fields_r1.intersection(fields_r2)
                    if common_fields:
                        common_df = pd.DataFrame(sorted(common_fields), columns=['Common Fields'])
                        st.dataframe(common_df, use_container_width=True)
                    else:
                        st.warning("No common fields found!")
                
                with tab2:
                    unique_r1 = fields_r1 - fields_r2
                    if unique_r1:
                        unique_r1_df = pd.DataFrame(sorted(unique_r1), columns=['Fields Only in R1'])
                        st.dataframe(unique_r1_df, use_container_width=True)
                        st.info(f"Found {len(unique_r1)} fields unique to R1")
                    else:
                        st.success("No unique fields in R1")
                
                with tab3:
                    unique_r2 = fields_r2 - fields_r1
                    if unique_r2:
                        unique_r2_df = pd.DataFrame(sorted(unique_r2), columns=['Fields Only in R2'])
                        st.dataframe(unique_r2_df, use_container_width=True)
                        st.info(f"Found {len(unique_r2)} fields unique to R2")
                    else:
                        st.success("No unique fields in R2")
            
            # === DATA COMPLETENESS ANALYSIS ===
            st.markdown("---")
            st.subheader("📊 Data Completeness Analysis")
            
            # Calculate completeness for each column
            completeness_ar1 = {}
            completeness_ar2 = {}
            
            for col in df_r1.columns:
                total_rows = len(df_r1)
                non_null_count = df_r1[col].count()
                completeness_ar1[col] = (non_null_count / total_rows) * 100
            
            for col in df_r2.columns:
                total_rows = len(df_r2)
                non_null_count = df_r2[col].count()
                completeness_ar2[col] = (non_null_count / total_rows) * 100
            
            # Create completeness comparison dataframe
            all_columns = set(df_r1.columns).union(set(df_r2.columns))
            completeness_comparison = []
            
            for col in sorted(all_columns):
                ar1_completeness = completeness_ar1.get(col, 0)
                ar2_completeness = completeness_ar2.get(col, 0)
                difference = ar1_completeness - ar2_completeness
                
                completeness_comparison.append({
                    'Column': col,
                    'R1_Completeness_%': round(ar1_completeness, 2),
                    'R2_Completeness_%': round(ar2_completeness, 2),
                    'Difference_%': round(difference, 2),
                    'Status': '✅ Better in R1' if difference > 5 else '✅ Better in R2' if difference < -5 else '🟰 Similar'
                })
            
            completeness_df = pd.DataFrame(completeness_comparison)
            
            # Display completeness summary
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_r1 = completeness_df['R1_Completeness_%'].mean()
                st.metric("R1 Avg Completeness", f"{avg_r1:.1f}%")
            with col2:
                avg_r2 = completeness_df['R2_Completeness_%'].mean()
                st.metric("R2 Avg Completeness", f"{avg_r2:.1f}%")
            with col3:
                overall_diff = avg_r1 - avg_r2
                st.metric("Overall Difference", f"{overall_diff:+.1f}%")
            
            st.dataframe(completeness_df, use_container_width=True)
            
            # === VALUE COMPARISON ANALYSIS ===
            if 'Field' in df_r1.columns and 'Field' in df_r2.columns:
                st.markdown("---")
                st.subheader("🔢 Value Comparison Analysis")
                
                # Merge dataframes on Field for comparison
                df_r1_clean = df_r1.copy()
                df_r2_clean = df_r2.copy()
                
                # Get numeric columns (excluding 'Field')
                numeric_cols_ar1 = df_r1_clean.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols_ar2 = df_r2_clean.select_dtypes(include=[np.number]).columns.tolist()
                common_numeric_cols = list(set(numeric_cols_ar1).intersection(set(numeric_cols_ar2)))
                
                if common_numeric_cols:
                    # Merge on Field
                    merged_df = pd.merge(df_r1_clean[['Field'] + common_numeric_cols], 
                                       df_r2_clean[['Field'] + common_numeric_cols], 
                                       on='Field', suffixes=('_R1', '_R2'), how='outer')
                    
                    # Calculate differences for each numeric column
                    value_comparison = []
                    for col in common_numeric_cols:
                        col_r1 = f"{col}_R1"
                        col_r2 = f"{col}_R2"
                        
                        if col_r1 in merged_df.columns and col_r2 in merged_df.columns:
                            # Calculate metrics
                            total_fields = len(merged_df)
                            both_have_data = merged_df[[col_r1, col_r2]].dropna().shape[0]
                            exact_matches = (merged_df[col_r1] == merged_df[col_r2]).sum()
                            
                            # Calculate average difference for non-null pairs
                            valid_pairs = merged_df[[col_r1, col_r2]].dropna()
                            if len(valid_pairs) > 0:
                                avg_diff = (valid_pairs[col_r1] - valid_pairs[col_r2]).abs().mean()
                                max_diff = (valid_pairs[col_r1] - valid_pairs[col_r2]).abs().max()
                            else:
                                avg_diff = None
                                max_diff = None
                            
                            match_rate = (exact_matches / both_have_data * 100) if both_have_data > 0 else 0
                            
                            value_comparison.append({
                                'Column': col,
                                'Fields_with_Both_Values': both_have_data,
                                'Exact_Matches': exact_matches,
                                'Match_Rate_%': round(match_rate, 2),
                                'Avg_Absolute_Difference': round(avg_diff, 2) if avg_diff is not None else 'N/A',
                                'Max_Absolute_Difference': round(max_diff, 2) if max_diff is not None else 'N/A'
                            })
                    
                    value_comparison_df = pd.DataFrame(value_comparison)
                    st.dataframe(value_comparison_df, use_container_width=True)
                    
                    # Show detailed mismatches for selected column
                    if not value_comparison_df.empty:
                        selected_col = st.selectbox("Select column to view detailed differences:", common_numeric_cols)
                        
                        if selected_col:
                            col_r1 = f"{selected_col}_R1"
                            col_r2 = f"{selected_col}_R2"
                            
                            # Show mismatches
                            mismatches = merged_df[
                                (merged_df[col_r1].notna()) & 
                                (merged_df[col_r2].notna()) & 
                                (merged_df[col_r1] != merged_df[col_r2])
                            ][['Field', col_r1, col_r2]].copy()
                            
                            if not mismatches.empty:
                                mismatches['Difference'] = mismatches[col_r1] - mismatches[col_r2]
                                mismatches['Abs_Difference'] = mismatches['Difference'].abs()
                                mismatches = mismatches.sort_values('Abs_Difference', ascending=False)
                                
                                st.subheader(f"🔍 Detailed Differences for {selected_col}")
                                st.dataframe(mismatches, use_container_width=True)
                            else:
                                st.success(f"✅ No differences found in {selected_col}")
            
            # === GENERATE COMPREHENSIVE REPORT ===
            st.markdown("---")
            st.subheader("📄 Generate Comprehensive Report")
            
            if st.button("🔄 Generate Analysis Report", type="primary"):
                # Create comprehensive report
                report = []
                report.append("# R1 vs R2 Extraction Comparison Report")
                report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report.append(f"R1 File: {r1_file.name}")
                report.append(f"R2 File: {r2_file.name}")
                report.append("")
                
                # Dataset Overview
                report.append("## Dataset Overview")
                report.append(f"- R1: {len(df_r1)} rows, {len(df_r1.columns)} columns")
                report.append(f"- R2: {len(df_r2)} rows, {len(df_r2.columns)} columns")
                report.append("")
                
                if 'Field' in df_r1.columns and 'Field' in df_r2.columns:
                    # Field Analysis
                    report.append("## Field Name Analysis")
                    report.append(f"- Fields in R1: {len(fields_r1)}")
                    report.append(f"- Fields in R2: {len(fields_r2)}")
                    report.append(f"- Common fields: {len(fields_r1.intersection(fields_r2))}")
                    report.append(f"- Field consistency rate: {consistency_rate:.1f}%")
                    report.append("")
                    
                    if unique_r1:
                        report.append(f"### Fields unique to R1 ({len(unique_r1)}):")
                        for field in sorted(unique_r1):
                            report.append(f"- {field}")
                        report.append("")
                    
                    if unique_r2:
                        report.append(f"### Fields unique to R2 ({len(unique_r2)}):")
                        for field in sorted(unique_r2):
                            report.append(f"- {field}")
                        report.append("")
                
                # Completeness Analysis
                report.append("## Data Completeness Analysis")
                report.append(f"- R1 average completeness: {avg_r1:.1f}%")
                report.append(f"- R2 average completeness: {avg_r2:.1f}%")
                report.append(f"- Overall difference: {overall_diff:+.1f}%")
                report.append("")
                
                # Key Findings
                report.append("## Key Findings")
                if consistency_rate >= 90:
                    report.append("✅ High field name consistency between extractions")
                elif consistency_rate >= 70:
                    report.append("⚠️ Moderate field name consistency - some discrepancies found")
                else:
                    report.append("❌ Low field name consistency - significant discrepancies found")
                
                if abs(overall_diff) <= 5:
                    report.append("✅ Similar data completeness between extractions")
                elif overall_diff > 5:
                    report.append("📈 R1 has better overall data completeness")
                else:
                    report.append("📈 R2 has better overall data completeness")
                
                report.append("")
                report.append("## Recommendations")
                if unique_r1 or unique_r2:
                    report.append("- Review field extraction logic to ensure consistency")
                if abs(overall_diff) > 10:
                    report.append("- Investigate data completeness differences")
                report.append("- Consider standardizing field naming conventions")
                
                # Convert report to string and offer download
                report_text = "\n".join(report)
                
                st.text_area("📋 Generated Report Preview", report_text, height=300)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Report (TXT)",
                        data=report_text,
                        file_name=f"AR_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Create Excel report with multiple sheets
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Summary sheet
                        if 'Field' in df_r1.columns and 'Field' in df_r2.columns:
                            summary_data = {
                                'Metric': ['R1 Rows', 'R1 Columns', 'R2 Rows', 'R2 Columns', 
                                          'Fields in R1', 'Fields in R2', 'Common Fields', 'Field Consistency %',
                                          'R1 Avg Completeness %', 'R2 Avg Completeness %', 'Completeness Difference %'],
                                'Value': [len(df_r1), len(df_r1.columns), len(df_r2), len(df_r2.columns),
                                         len(fields_r1), len(fields_r2), len(fields_r1.intersection(fields_r2)), 
                                         round(consistency_rate, 2), round(avg_r1, 2), round(avg_r2, 2), round(overall_diff, 2)]
                            }
                            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Completeness comparison
                        completeness_df.to_excel(writer, sheet_name='Completeness_Comparison', index=False)
                        
                        # Value comparison if available
                        if 'value_comparison_df' in locals() and not value_comparison_df.empty:
                            value_comparison_df.to_excel(writer, sheet_name='Value_Comparison', index=False)
                    
                    excel_data = output.getvalue()
                    st.download_button(
                        label="📊 Download Detailed Report (Excel)",
                        data=excel_data,
                        file_name=f"AR_comparison_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            st.success("✅ Analysis complete! Use the buttons above to download comprehensive reports.")
            
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.info("Please ensure your files have the expected format with at least a 'Field' column for comparison.")
    
    else:
        st.info("⬆️ Please upload both R1 and R2 files to begin analysis.")
        
        with st.expander("📖 Expected File Format"):
            st.markdown("""
            **Both files should contain:**
            - A 'Field' column with field names
            - Numeric columns with extracted values
            - Consistent structure for meaningful comparison
            
            **Example format:**
            ```
            Field,2021,2022,2023
            Total Revenue,1000000,1100000,1200000
            Operating Expenses,800000,850000,900000
            Net Income,200000,250000,300000
            ```
            """)