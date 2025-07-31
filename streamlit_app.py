import streamlit as st
import pandas as pd

st.set_page_config(page_title="Extraction Accuracy Checker", layout="wide")
st.title("📊 Financial Statement Accuracy Checker")

# Upload files
gt_file = st.file_uploader("📥 Upload Ground Truth CSV", type="csv")
extract_file = st.file_uploader("📥 Upload Extracted File (Excel or CSV)", type=["xlsx", "csv"])

# Helper functions (same as your original code)
def normalize(field):
    return str(field).strip().lower()

def to_float(val):
    try:
        return float(str(val).replace(",", "").replace("(", "-").replace(")", "").replace("$", ""))
    except:
        return None

if gt_file and extract_file:
    try:
        # Load data
        gt_df = pd.read_csv(gt_file)
        
        # Handle both Excel and CSV files
        if extract_file.name.endswith('.csv'):
            df = pd.read_csv(extract_file)
        else:
            # Handle Excel file with multiple sheets (use first sheet like your code)
            xls = pd.ExcelFile(extract_file)
            df = xls.parse(xls.sheet_names[0])
        
        # Prepare Ground Truth data (following your exact logic)
        gt_long = gt_df.melt(id_vars=['Category','Field'], var_name='Year', value_name='Value_GT')
        gt_long['Year'] = gt_long['Year'].astype(str).str.strip()
        gt_long['Field'] = gt_long['Field'].str.strip()
        
        # Prepare Extracted data (following your exact logic)
        df['Normalized_Field'] = df['Field'].apply(lambda x: str(x).replace(" Current Year", "").replace(" Prior Year", "").strip())
        
        df_long = df.melt(id_vars='Normalized_Field', var_name='Year', value_name='Value_Extracted')
        df_long['Year'] = df_long['Year'].astype(str).str.strip()
        df_long['Normalized_Field'] = df_long['Normalized_Field'].str.strip()
        
        # Apply full normalization (your exact logic)
        gt_long['Field'] = gt_long['Field'].apply(normalize)
        df_long['Normalized_Field'] = df_long['Normalized_Field'].apply(normalize)
        
        # Merge data (your exact logic)
        comparison_df = pd.merge(
            df_long,
            gt_long,
            left_on=['Normalized_Field', 'Year'],
            right_on=['Field', 'Year'],
            how='outer'
        )
        
        # Convert values and compare (your exact logic)
        comparison_df['Value_Extracted'] = comparison_df['Value_Extracted'].apply(to_float)
        comparison_df['Value_GT'] = comparison_df['Value_GT'].apply(to_float)
        comparison_df['Match'] = comparison_df['Value_Extracted'] == comparison_df['Value_GT']
        
        # Calculate accuracy (your exact logic)
        total = comparison_df['Value_GT'].notna().sum()
        correct = comparison_df['Match'].sum()
        accuracy = correct / total if total else 0
        
        # Display results
        st.metric("✅ Extraction Accuracy", f"{accuracy:.2%}", f"{correct}/{total} fields matched")
        
        # Show mismatches (your exact logic)
        st.subheader("🔍 Mismatched Fields")
        mismatches = comparison_df[~comparison_df['Match']][['Normalized_Field', 'Year', 'Value_Extracted', 'Value_GT','Category']]
        
        if len(mismatches) > 0:
            st.dataframe(mismatches, use_container_width=True)
        else:
            st.success("🎉 No mismatches found!")
        
        # Export functionality (your exact logic)
        csv = comparison_df.to_csv(index=False).encode()
        st.download_button(
            "📤 Download Full Comparison CSV", 
            csv, 
            "full_comparison_results.csv", 
            "text/csv"
        )
        
        # Success message
        st.success("📋 Analysis complete! You can download the full results above.")
        
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
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
    
    # Show example of expected format
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
        
        Note: The code automatically removes " Current Year" and " Prior Year" from field names during processing.
        """)