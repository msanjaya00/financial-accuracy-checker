import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
from difflib import SequenceMatcher
import re
from collections import defaultdict

st.set_page_config(page_title="Composite Key Financial Accuracy Checker", layout="wide")
st.title("🎯 Composite Key Financial Accuracy Checker")

def preprocess_field_name(field):
    """Clean and standardize field names for better matching"""
    if pd.isna(field) or field == "":
        return ""
    
    field_str = str(field).lower().strip()
    
    # Remove common prefixes/suffixes
    field_str = re.sub(r'\b(current|prior)\s+(year|period)\b', '', field_str)
    field_str = re.sub(r'\b(year|period)\s+(ended?|ending)\b', '', field_str)
    field_str = re.sub(r'\b(as\s+of|at)\b', '', field_str)
    
    # Standardize plurals and common terms
    replacements = {
        'assets': 'asset', 'liabilities': 'liability', 'revenues': 'revenue',
        'expenses': 'expense', 'receivables': 'receivable', 'payables': 'payable',
        'inventories': 'inventory', 'securities': 'security', 'stockholders': 'stockholder'
    }
    
    for old, new in replacements.items():
        field_str = re.sub(r'\b' + old + r'\b', new, field_str)
    
    # Clean punctuation and spaces
    field_str = re.sub(r'[^\w\s]', ' ', field_str)
    field_str = re.sub(r'\s+', ' ', field_str).strip()
    
    # Remove stop words
    stop_words = {'of', 'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from'}
    words = [word for word in field_str.split() if word not in stop_words]
    
    return ' '.join(words)

def parse_keywords(keywords_string):
    """Parse keywords from string"""
    if pd.isna(keywords_string) or keywords_string == "":
        return []
    
    keywords_clean = str(keywords_string).replace('"', '').replace("'", "")
    if ';' in keywords_clean:
        keywords = [kw.strip().lower() for kw in keywords_clean.split(';')]
    else:
        keywords = [kw.strip().lower() for kw in keywords_clean.split(',')]
    
    return [kw for kw in keywords if kw]

def calculate_keyword_similarity(extracted_field, keywords):
    """Calculate similarity based on keyword matching with directional awareness"""
    if not keywords or pd.isna(extracted_field) or extracted_field == "":
        return 0.0
    
    field_clean = preprocess_field_name(extracted_field).lower()
    field_words = set(field_clean.split())
    
    # Check for directional conflicts (Due From vs Due To)
    if 'due' in field_clean:
        field_has_from = 'from' in field_clean
        field_has_to = 'to' in field_clean or 'due to' in extracted_field.lower()
        
        for keyword in keywords:
            keyword_clean = keyword.strip().lower()
            if 'due' in keyword_clean:
                keyword_has_from = 'from' in keyword_clean
                keyword_has_to = 'to' in keyword_clean
                
                # Penalize opposite directions
                if (field_has_from and keyword_has_to) or (field_has_to and keyword_has_from):
                    return 0.1  # Very low score for opposite directions
    
    exact_matches = 0
    partial_matches = 0
    
    for keyword in keywords:
        keyword = keyword.strip().lower()
        if keyword in field_clean:
            exact_matches += 1
        elif any(word in keyword or keyword in word for word in field_words if len(word) > 2):
            partial_matches += 1
    
    if len(keywords) == 0:
        return 0.0
    
    # Weight exact matches more heavily
    similarity = (exact_matches * 1.0 + partial_matches * 0.6) / len(keywords)
    return min(similarity, 1.0)

def calculate_field_similarity(field1, field2):
    """Calculate direct field name similarity with directional awareness"""
    if pd.isna(field1) or pd.isna(field2) or field1 == "" or field2 == "":
        return 0.0
    
    clean1 = preprocess_field_name(field1)
    clean2 = preprocess_field_name(field2)
    
    # Check for directional conflicts (Due From vs Due To)
    if 'due' in clean1.lower() and 'due' in clean2.lower():
        field1_has_from = 'from' in clean1.lower()
        field1_has_to = 'to' in clean1.lower()
        field2_has_from = 'from' in clean2.lower()
        field2_has_to = 'to' in clean2.lower()
        
        # Penalize opposite directions
        if (field1_has_from and field2_has_to) or (field1_has_to and field2_has_from):
            return 0.1  # Very low score for opposite directions
    
    if clean1.lower() == clean2.lower():  # Case insensitive exact match
        return 1.0
    
    # Sequence matching
    seq_similarity = SequenceMatcher(None, clean1.lower(), clean2.lower()).ratio()
    
    # Word-based similarity (case insensitive)
    words1 = set(clean1.lower().split())
    words2 = set(clean2.lower().split())
    if len(words1.union(words2)) > 0:
        word_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
    else:
        word_similarity = 0.0
    
    return (0.6 * seq_similarity + 0.4 * word_similarity)

def detect_statement_type(field_name):
    """Detect statement type from field name"""
    if field_name and isinstance(field_name, str) and '_' in field_name:
        prefix = field_name.split('_')[0].upper()
        if prefix in ['BS', 'IS', 'RE', 'CF', 'FN']:
            return prefix
    return 'UNKNOWN'

def calculate_composite_similarity(extracted_field, composite_key, gt_field, keywords):
    """Calculate comprehensive similarity with directional awareness and case insensitivity"""
    # Keyword similarity (primary)
    keyword_score = calculate_keyword_similarity(extracted_field, keywords)
    
    # Field name similarity (secondary)
    field_score = calculate_field_similarity(extracted_field, gt_field)
    
    # Composite key structure bonus (case insensitive)
    composite_bonus = 0.0
    if composite_key and '_' in composite_key:
        composite_parts = composite_key.split('_')[1:]  # Remove prefix like BS_, IS_
        extracted_lower = extracted_field.lower()
        
        # Check if composite key parts match extracted field (case insensitive)
        composite_text = ' '.join(composite_parts).lower().replace('_', ' ')
        if composite_text in extracted_lower or extracted_lower in composite_text:
            composite_bonus = 0.15
        elif any(part.lower() in extracted_lower for part in composite_parts if len(part) > 2):
            composite_bonus = 0.1
    
    # Special handling for exact matches (case insensitive)
    if extracted_field.lower().strip() == gt_field.lower().strip():
        return 0.95  # High confidence for exact matches
    
    # Weighted combination (prioritize keywords but boost field matches)
    if field_score > 0.8:  # High field similarity gets extra weight
        final_similarity = (0.4 * keyword_score + 0.5 * field_score + 0.1 * composite_bonus)
    else:
        final_similarity = (0.6 * keyword_score + 0.3 * field_score + 0.1 * composite_bonus)
    
    return min(final_similarity, 1.0)

def find_best_matches(gt_df, extracted_fields, similarity_threshold=0.5):
    """Find best matches using composite key intelligence"""
    matches = {}
    match_scores = {}
    match_details = {}
    
    # Prepare ground truth data
    gt_data = {}
    for _, row in gt_df.iterrows():
        composite_key = row.get('Composite Key', '')
        field = row.get('Field', '')
        keywords = parse_keywords(row.get('Keywords', ''))
        
        gt_data[composite_key] = {
            'field': field,
            'keywords': keywords,
            'statement_type': detect_statement_type(composite_key)
        }
    
    # Calculate similarities
    similarities = {}
    for composite_key, gt_info in gt_data.items():
        similarities[composite_key] = {}
        for ext_field in extracted_fields:
            similarity = calculate_composite_similarity(
                ext_field, composite_key, gt_info['field'], gt_info['keywords']
            )
            similarities[composite_key][ext_field] = similarity
    
    # Find best matches using greedy approach
    unmatched_gt = set(gt_data.keys())
    unmatched_extracted = set(extracted_fields)
    
    while True:
        best_score = similarity_threshold
        best_gt = None
        best_ext = None
        
        for composite_key in unmatched_gt:
            for ext_field in unmatched_extracted:
                score = similarities[composite_key][ext_field]
                if score > best_score:
                    best_score = score
                    best_gt = composite_key
                    best_ext = ext_field
        
        if best_gt is None:
            break
        
        matches[best_gt] = best_ext
        match_scores[best_gt] = best_score
        
        # Store match details
        gt_info = gt_data[best_gt]
        match_details[best_gt] = {
            'keyword_score': calculate_keyword_similarity(best_ext, gt_info['keywords']),
            'field_score': calculate_field_similarity(best_ext, gt_info['field']),
            'keywords': gt_info['keywords'][:3]  # First 3 keywords
        }
        
        unmatched_gt.remove(best_gt)
        unmatched_extracted.remove(best_ext)
    
    return matches, match_scores, match_details, unmatched_gt, unmatched_extracted

def to_float(val):
    """Convert value to float"""
    if pd.isna(val) or val == "" or val is None:
        return None
    try:
        val_str = str(val).replace(",", "").replace("(", "-").replace(")", "").replace("$", "").strip()
        return float(val_str) if val_str != "" else None
    except:
        return None

def calculate_percentage_diff(val1, val2):
    """Calculate percentage difference"""
    if pd.isna(val1) or pd.isna(val2) or val1 is None or val2 is None:
        return None
    if val2 == 0:
        return np.inf if val1 != 0 else 0
    return ((val1 - val2) / abs(val2)) * 100

def values_match(val1, val2, tolerance=1.0):
    """Check if values match within tolerance"""
    try:
        if pd.isna(val1) and pd.isna(val2):
            return True
        if pd.isna(val1) or pd.isna(val2):
            return False
        return abs(val1 - val2) <= tolerance
    except:
        return False

# Main application
st.markdown("### 🔧 Configuration")
col1, col2 = st.columns(2)
with col1:
    similarity_threshold = st.slider("Similarity Threshold", 0.3, 1.0, 0.5, 0.05)
with col2:
    tolerance = st.number_input("Value Tolerance", min_value=0.01, value=1.0, step=0.1)

# File uploads
st.markdown("### 📁 File Upload")
col1, col2 = st.columns(2)
with col1:
    gt_file = st.file_uploader("Upload Ground Truth CSV", type="csv", key="gt")
    if gt_file:
        st.success(f"✅ {gt_file.name}")

with col2:
    extract_file = st.file_uploader("Upload Extracted Results", type=["xlsx", "csv"], key="extract")
    if extract_file:
        st.success(f"✅ {extract_file.name}")

if gt_file and extract_file:
    try:
        # Load data
        gt_df = pd.read_csv(gt_file)
        
        # Validate ground truth structure
        required_cols = ['Field', 'Composite Key', 'Keywords']
        missing_cols = [col for col in required_cols if col not in gt_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()

        # Load extracted data
        if extract_file.name.endswith('.csv'):
            df = pd.read_csv(extract_file)
        else:
            xls = pd.ExcelFile(extract_file)
            if len(xls.sheet_names) > 1:
                selected_sheet = st.selectbox("Select sheet:", xls.sheet_names)
                df = xls.parse(selected_sheet)
            else:
                df = xls.parse(xls.sheet_names[0])

        if 'Field' not in df.columns:
            st.error("Extracted file missing 'Field' column")
            st.stop()

        # Data overview
        st.markdown("### 📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("GT Fields", len(gt_df))
        with col2:
            st.metric("Extracted Fields", len(df))
        with col3:
            statement_types = gt_df['Doc_ID'].nunique() if 'Doc_ID' in gt_df.columns else 0
            st.metric("Statement Types", statement_types)
        with col4:
            year_cols = [col for col in gt_df.columns if col.isdigit()]
            st.metric("Years", len(year_cols))

        # Find matches
        extracted_fields = df['Field'].dropna().unique().tolist()
        
        with st.spinner("🔍 Finding matches using composite keys..."):
            matches, match_scores, match_details, unmatched_gt, unmatched_extracted = find_best_matches(
                gt_df, extracted_fields, similarity_threshold
            )

        # Results summary
        st.markdown("### 🎯 Matching Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Matches Found", len(matches))
        with col2:
            match_rate = (len(matches) / len(gt_df)) * 100 if len(gt_df) > 0 else 0
            st.metric("Match Rate", f"{match_rate:.1f}%")
        with col3:
            avg_score = np.mean(list(match_scores.values())) if match_scores else 0
            st.metric("Avg Score", f"{avg_score:.3f}")

        # Show matches
        if matches:
            st.markdown("### ✅ Field Matches")
            match_display = []
            for composite_key, ext_field in matches.items():
                gt_row = gt_df[gt_df['Composite Key'] == composite_key].iloc[0]
                score = match_scores[composite_key]
                details = match_details[composite_key]
                
                confidence = "🟢 High" if score > 0.7 else "🟡 Medium" if score > 0.5 else "🔴 Low"
                keywords_display = ", ".join(details['keywords']) if details['keywords'] else ""
                
                match_display.append({
                    'Composite Key': composite_key,
                    'GT Field': gt_row['Field'],
                    'Extracted Field': ext_field,
                    'Score': f"{score:.3f}",
                    'Confidence': confidence,
                    'Top Keywords': keywords_display
                })
            
            match_df = pd.DataFrame(match_display)
            st.dataframe(match_df, use_container_width=True)

        # Analysis button
        if st.button("🚀 Run Accuracy Analysis", type="primary"):
            # Prepare data for analysis
            field_mapping = matches
            year_cols = [col for col in gt_df.columns if col.isdigit()]
            
            if not year_cols:
                st.error("No year columns found")
                st.stop()

            # Create long format for ground truth
            gt_long = gt_df.melt(
                id_vars=['Field', 'Composite Key'], 
                value_vars=year_cols,
                var_name='Year', 
                value_name='Value_GT'
            )
            gt_long['Year'] = gt_long['Year'].astype(str)

            # Map extracted data
            df_mapped = df.copy()
            df_mapped['Mapped_Key'] = df_mapped['Field'].map(
                lambda x: next((k for k, v in field_mapping.items() if v == x), None)
            )
            df_mapped = df_mapped[df_mapped['Mapped_Key'].notna()]

            # Create long format for extracted (preserve original field names)
            extract_year_cols = [col for col in df_mapped.columns if col.isdigit()]
            df_long = df_mapped.melt(
                id_vars=['Field', 'Mapped_Key'], 
                value_vars=extract_year_cols,
                var_name='Year', 
                value_name='Value_Extracted'
            )
            df_long['Year'] = df_long['Year'].astype(str)
            df_long['Original_Extracted_Field'] = df_long['Field']  # Preserve original

            # Merge for comparison
            comparison_df = pd.merge(
                df_long, gt_long,
                left_on=['Mapped_Key', 'Year'],
                right_on=['Composite Key', 'Year'],
                how='inner'
            )

            if comparison_df.empty:
                st.error("No data for comparison")
                st.stop()

            # Convert values
            comparison_df['Value_Extracted'] = comparison_df['Value_Extracted'].apply(to_float)
            comparison_df['Value_GT'] = comparison_df['Value_GT'].apply(to_float)
            
            # Calculate matches and differences
            comparison_df['Match'] = comparison_df.apply(
                lambda row: values_match(row['Value_Extracted'], row['Value_GT'], tolerance), axis=1
            )
            comparison_df['Percentage_Diff'] = comparison_df.apply(
                lambda row: calculate_percentage_diff(row['Value_Extracted'], row['Value_GT']), axis=1
            )

            # Results
            st.markdown("### 📈 Accuracy Results")
            
            valid_gt = comparison_df['Value_GT'].notna()
            total_fields = valid_gt.sum()
            
            if total_fields > 0:
                matches_count = comparison_df.loc[valid_gt, 'Match'].sum()
                accuracy = (matches_count / total_fields) * 100
                
                extracted_coverage = comparison_df.loc[valid_gt, 'Value_Extracted'].notna().sum()
                coverage = (extracted_coverage / total_fields) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.1f}%", f"{matches_count}/{total_fields}")
                with col2:
                    st.metric("Coverage", f"{coverage:.1f}%", f"{extracted_coverage}/{total_fields}")
                with col3:
                    valid_diffs = comparison_df['Percentage_Diff'].dropna()
                    mape = valid_diffs.abs().mean() if not valid_diffs.empty else 0
                    st.metric("MAPE", f"{mape:.2f}%")

            # Field-level results
            st.markdown("### 📊 Field-Level Results")
            field_results = []
            for composite_key in comparison_df['Composite Key'].unique():
                field_data = comparison_df[
                    (comparison_df['Composite Key'] == composite_key) & 
                    comparison_df['Value_GT'].notna()
                ]
                
                if len(field_data) > 0:
                    field_matches = field_data['Match'].sum()
                    field_total = len(field_data)
                    field_accuracy = (field_matches / field_total) * 100
                    
                    gt_row = gt_df[gt_df['Composite Key'] == composite_key].iloc[0]
                    
                    field_results.append({
                        'Composite Key': composite_key,
                        'Original Extracted Field': field_data['Original_Extracted_Field'].iloc[0] if 'Original_Extracted_Field' in field_data.columns else '',
                        'GT Field': gt_row['Field'],
                        'Accuracy %': round(field_accuracy, 1),
                        'Matches': f"{field_matches}/{field_total}",
                        'Match Score': f"{match_scores.get(composite_key, 0):.3f}"
                    })
            
            if field_results:
                field_df = pd.DataFrame(field_results)
                field_df = field_df.sort_values('Accuracy %', ascending=False)
                st.dataframe(field_df, use_container_width=True)

            # Mismatches with clear field traceability
            mismatches = comparison_df[~comparison_df['Match'] & comparison_df['Value_GT'].notna()]
            if not mismatches.empty:
                st.markdown("### ❌ Mismatches")
                
                # Create clear mismatch display with original and mapped fields
                mismatch_display = mismatches[[
                    'Composite Key', 'Original_Extracted_Field', 'Field_y', 'Year', 
                    'Value_Extracted', 'Value_GT', 'Percentage_Diff'
                ]].copy()
                
                # Rename columns for clarity
                mismatch_display.columns = [
                    'Composite Key', 'Original Extracted Field', 'GT Field Name', 'Year',
                    'Value_Extracted', 'Value_GT', 'Percentage_Diff'
                ]
                
                if 'Percentage_Diff' in mismatch_display.columns:
                    mismatch_display = mismatch_display.sort_values(
                        'Percentage_Diff', key=lambda x: x.abs(), ascending=False, na_position='last'
                    )
                
                st.dataframe(mismatch_display, use_container_width=True)
                
                # Summary of mismatch patterns
                st.markdown("#### 🔍 Mismatch Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Mismatches", len(mismatches))
                    avg_error = mismatches['Percentage_Diff'].abs().mean() if not mismatches['Percentage_Diff'].empty else 0
                    st.metric("Avg Error %", f"{avg_error:.2f}%")
                
                with col2:
                    # Show fields with most mismatches
                    field_mismatch_counts = mismatches['Composite Key'].value_counts()
                    if not field_mismatch_counts.empty:
                        st.write("**Most Problematic Fields:**")
                        for field, count in field_mismatch_counts.head(3).items():
                            original_field = mismatches[mismatches['Composite Key'] == field]['Original_Extracted_Field'].iloc[0]
                            st.write(f"• {field}: {count} mismatches")
                            st.write(f"  ↳ *Original: {original_field}*")

            # Export
            st.markdown("### 📤 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = comparison_df.to_csv(index=False).encode()
                st.download_button(
                    "📄 Download CSV",
                    csv_data,
                    f"accuracy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            
            with col2:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Summary
                    summary = pd.DataFrame([
                        ['Total Fields', len(gt_df)],
                        ['Matched Fields', len(matches)],
                        ['Match Rate %', round(match_rate, 1)],
                        ['Accuracy %', round(accuracy, 1) if 'accuracy' in locals() else 0],
                        ['Coverage %', round(coverage, 1) if 'coverage' in locals() else 0]
                    ], columns=['Metric', 'Value'])
                    summary.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Field mapping
                    if match_df is not None:
                        match_df.to_excel(writer, sheet_name='Field_Mapping', index=False)
                    
                    # Full comparison
                    comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
                    
                    # Field results
                    if field_results:
                        field_df.to_excel(writer, sheet_name='Field_Results', index=False)
                
                excel_data = output.getvalue()
                st.download_button(
                    "📊 Download Excel",
                    excel_data,
                    f"accuracy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("📁 Upload both Ground Truth CSV and Extracted Results file to begin")
    
    with st.expander("📖 Expected File Formats"):
        st.markdown("""
        **Ground Truth CSV:**
        - **Composite Key**: BS_Total_Assets, IS_Net_Income, CF_Operating_Cash_Flow
        - **Field**: Original field names  
        - **Keywords**: Semicolon-separated keywords for matching
        - **Year columns**: 2021, 2022, 2023, 2024 (with values)
        
        **Extracted Results (CSV/Excel):**
        - **Field**: Extracted field names
        - **Year columns**: Same years with extracted values
        
        **Example:**
        ```
        Composite Key,Field,Keywords,2024,2023
        BS_Total_Assets,"Total Assets","total;asset;sum;balance",1000000,950000
        IS_Net_Income,"Net Income","net;income;profit;earnings",50000,45000
        ```
        """)

# Footer
st.markdown("---")
st.markdown("**🎯 Composite Key Financial Accuracy Checker • Powered by intelligent keyword matching**")