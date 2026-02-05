import streamlit as st
import pandas as pd
import numpy as np
import gspread
import re
from fpdf import FPDF
from google.oauth2.service_account import Credentials


# --- PASTE THIS BLOCK AT THE TOP ---
def add_total_row(df):
    if df.empty: return df
    
    # Select only number columns to sum
    numeric_df = df.select_dtypes(include=['number'])
    total_row = pd.DataFrame(numeric_df.sum()).T
    
    # Add back missing text columns (empty)
    for col in df.columns:
        if col not in total_row.columns:
            total_row[col] = ""
            
    # Set the Name to "TOTAL" (No exclamation marks)
    if 'Name' in total_row.columns:
        total_row['Name'] = "TOTAL"  # <--- CHANGED HERE
    else:
        total_row.iloc[0, 0] = "TOTAL" # <--- AND HERE
        
    # Reorder columns
    total_row = total_row[df.columns]
    
    return pd.concat([df, total_row], ignore_index=True)
# -----------------------------------

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Galaxy Junket Management System", layout="wide", page_icon="🎰")


# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .invoice-container {
        background-color: white;
        padding: 40px;
        border: 1px solid #ddd;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        max-width: 900px;
        margin: auto;
    }
</style>
""", unsafe_allow_html=True)

def draw_box(label, value, suffix="HKD", from_percent=False):

    # FORCE integer display everywhere
    if from_percent:
        display_value = f"{int(value):,}"      # drop decimals
    else:
        display_value = f"{int(round(value)):,}"  # normal rounding, no .0

    val_color = "#d9534f" if value < 0 else "#212529"

    st.markdown(f"""
    <div style="
        background-color: white;
        border-left: 10px solid #0056b3;
        border-top: 1px solid #dee2e6;
        border-right: 1px solid #dee2e6;
        border-bottom: 1px solid #dee2e6;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        height: 120px;
    ">
        <div style="color: #666; font-size: 14px; font-weight: bold; margin-bottom: 8px;">
            {label}
        </div>
        <div style="color: {val_color}; font-size: 26px; font-weight: bold;">
            {display_value}
            <span style="font-size: 16px; font-weight: normal;">
                {suffix}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.title("🎰 Galaxy Junket Management System")

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

# --- SIDEBAR ---
st.sidebar.subheader("Data Source")
st.sidebar.header("📂 Data & Settings")
hkd_to_thb = st.sidebar.number_input("HKD to THB Rate", value=6.00)

# 🔄 Refresh Now — BOTTOM of sidebar
st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

# --- FUNCTIONS ---
import re # PUT THIS AT THE VERY TOP OF YOUR FILE IF MISSING

@st.cache_data(ttl=60)
def load_and_map_df(raw: pd.DataFrame):    
    # 1. Clean headers: remove newlines, extra spaces
    raw.columns = raw.columns.astype(str).str.replace('\n', ' ').str.strip()
    clean_df = pd.DataFrame()

    # 2. Smart Column Finder
    def find_col(exact_name, keywords, exclude_keywords=None):
        if exact_name in raw.columns: return exact_name
        for actual in raw.columns:
            if all(k.lower() in actual.lower() for k in keywords):
                if exclude_keywords and any(ek.lower() in actual.lower() for ek in exclude_keywords):
                    continue 
                return actual
        return None

    # --- MAPPING COLUMNS ---
    c_name = find_col("Player Name", ["name"]) 
    c_turn = find_col("Turnover HKD", ["turnover"])
    c_wl   = find_col("Gross Player Win/loss HKD", ["gross", "win"])
    c_mon  = find_col("Month_Name", ["month"])
    
    # Expenses (Separate Player vs Junket)
    c_pexp = find_col("Player Expense", ["player", "expense"]) 
    c_jexp = find_col("Expense", ["expense"], exclude_keywords=["player"]) 

    # Payments (Flexible Search)
    c_pby  = find_col("Paid by Player", ["paid", "by"]) 
    if not c_pby: c_pby = find_col("Paid by Player", ["payment", "in"])
    
    c_pto  = find_col("Paid to Player", ["paid", "to"])
    if not c_pto: c_pto = find_col("Paid to Player", ["payment", "out"])

    c_pcm  = find_col("Player Commission Amount HKD", ["commission"])
    c_bon  = find_col("Month end com Bonus HKD", ["bonus"])

    # --- NUMBER CLEANER (Removes $, HKD, commas) ---
    def clean_number(val):
        s = str(val).strip()
        if not s: return 0.0
        # Handle (100) -> -100
        if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
        # Remove anything that isn't a number or dot
        s = re.sub(r'[^\d\.-]', '', s)
        try: return float(s)
        except: return 0.0

    def to_num(col_name):
        if col_name and col_name in raw.columns:
            return raw[col_name].apply(clean_number).fillna(0)
        return pd.Series(0.0, index=raw.index)

    # --- BUILD CLEAN DATAFRAME ---
    clean_df['Name'] = raw[c_name].astype(str).str.strip() if c_name else "Unknown"
    clean_df['Month_Name'] = raw[c_mon].astype(str).str.strip() if c_mon else "Unknown"
    
    clean_df['Turnover'] = to_num(c_turn)
    clean_df['Gross Player Win/Loss'] = to_num(c_wl)
    
    # --- EXPENSES (CRITICAL FIX FOR TAB 1) ---
    clean_df['Junket Expenses'] = to_num(c_jexp)
    clean_df['Player Expenses'] = to_num(c_pexp)
    # This prevents Tab 1 from crashing (KeyError):
    clean_df['Expenses'] = clean_df['Junket Expenses'] 
    
    clean_df['Paid by Player'] = to_num(c_pby)
    clean_df['Paid to Player'] = to_num(c_pto)
    clean_df['Player Comm'] = to_num(c_pcm)
    clean_df['Bonus'] = np.floor(to_num(c_bon))

    return clean_df
# --- UNIVERSAL CONNECTION (Cloud + Laptop) ---
def load_from_gsheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly"
    ]

    # We use 'try' to safely check for Cloud Secrets
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scopes
        )
    except:
        # If Secrets fail (because we are on a laptop), use the file instead
        # MAKE SURE 'sheets-reader.json' IS IN YOUR FOLDER!
        creds = Credentials.from_service_account_file(
            "sheets-reader.json",
            scopes=scopes
        )

    client = gspread.authorize(creds)

    sheet = client.open_by_url(
        "https://docs.google.com/spreadsheets/d/1AXFqf2IR3KUpxMskKSeOcqXXbOaU0_Hc0eiGp8xcEGA/edit"
    ).worksheet("DATABASE")

    records = sheet.get_all_records()
    return pd.DataFrame(records)

# --- LOAD DATA ---
with st.spinner("🔄 Loading data..."):
    raw_df = load_from_gsheet()   
    df = load_and_map_df(raw_df)

st.success("✅ Data loaded")


# --- CALCULATIONS ---
df['Junket Comm'] = np.floor(df['Turnover'] * 0.0115)

# Update Junket Net to use JUNKET Expenses
df['Total Comm Payout'] = df['Junket Comm'] + df['Bonus'] - df['Junket Expenses']
df['Junket Net HKD'] = df['Total Comm Payout'] - df['Player Comm']
df['Junket Net THB'] = (df['Junket Net HKD'] * hkd_to_thb).round(0)

# Update Player Net to use PLAYER Expenses
# Formula: Gross Win + Commission - Player Expenses
df['Player Net HKD'] = (df['Gross Player Win/Loss'] + df['Player Comm'] - df['Player Expenses']).round(0)
df['Player Net THB'] = (df['Player Net HKD'] * hkd_to_thb).round(0)

# --- FILTERS ---
month_options = sorted(df['Month_Name'].unique())

sel_months = st.sidebar.multiselect(
    "Month",
    ["Select All"] + month_options,
    default=["Select All"]
)

if "Select All" in sel_months:
    sel_months = month_options

players = sorted([p for p in df['Name'].unique() if str(p).lower() != 'nan'])

# add Select All option
player_options = ["Select All"] + players

sel_players = st.sidebar.multiselect(
    "Select Player Name",
    player_options,
    default=["Select All"]
)

# handle Select All logic
if "Select All" in sel_players:
    sel_players = players


df_f = df[df['Month_Name'].isin(sel_months) & df['Name'].isin(sel_players)]
df_pool = df[df['Month_Name'].isin(sel_months)]

tab1, tab2, tab3 = st.tabs(["🏢 Junket Dashboard", "👥 Player Report", "🧾 Individual Invoice"])

def highlight_negative(val):
    try:
        val = float(val)
        color = 'red' if val < 0 else 'black'
        return f'color: {color}'
    except:
        return ''

with tab1:
    st.subheader("🏢 Junket Main Summary")

    # 1. METRIC BOXES
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        draw_box("Turnover in HKD", df_f['Turnover'].sum())
    with r1c2:
        draw_box("Junket Comm (1.15%)", df_f['Junket Comm'].sum())
    with r1c3:
        draw_box("Month end Bonus", df_f['Bonus'].sum())
    with r1c4:
        draw_box("Expense", df_f['Expenses'].sum())

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        draw_box("Total Comm Payout", df_f['Total Comm Payout'].sum())
    with r2c2:
        draw_box("Player Comm HKD", df_f['Player Comm'].sum())
    with r2c3:
        draw_box("Junket Net Total HKD", df_f['Junket Net HKD'].sum())
    with r2c4:
        total_thb = int(df_f['Junket Net HKD'].sum() * hkd_to_thb)
        draw_box("Junket Net Total THB", total_thb, "THB")

    st.divider()

    # 2. STYLING FUNCTIONS (Defined locally to avoid scope issues)
    def highlight_negative(val):
        try:
            val = float(val)
            return 'color: #d9534f' if val < 0 else 'color: #212529'
        except:
            return ''

    def highlight_red(val):
        return 'color: #d9534f'
        
    def bold_total(row):
        if row['Name'] == 'TOTAL':
            return ['font-weight: bold; background-color: #f0f0f0'] * len(row)
        return [''] * len(row)

    # 3. DISPLAY TABLE
    df_with_total = add_total_row(df_f)
    exp_col = 'Expenses' if 'Expenses' in df_with_total.columns else 'Junket Expenses'
    
    st.dataframe(
        df_with_total.style
        .format("{:,.0f}", subset=df_with_total.select_dtypes("number").columns)
        .applymap(highlight_negative) 
        .applymap(highlight_red, subset=[exp_col] if exp_col in df_with_total.columns else None)
        .apply(bold_total, axis=1),
        use_container_width=True,
        hide_index=True
    )

    # 4. DOWNLOAD BUTTON (Now perfectly aligned)
    csv_tab1 = convert_df(df_with_total)
    
    st.download_button(
        label="📥 Download Junket Summary (CSV)",
        data=csv_tab1,
        file_name='Junket_Summary_Report.csv',
        mime='text/csv'
    )
with tab2:
    # 1. HEADER
    st.subheader("👥 PLAYER SUMMARY REPORT")

    # 2. METRIC BOXES
    pk1, pk2, pk3 = st.columns(3)
    with pk1:
        draw_box("Turnover in HKD", df_f['Turnover'].sum())
    with pk2:
        draw_box("Gross Player Win/Loss HKD", df_f['Gross Player Win/Loss'].sum())
    with pk3:
        draw_box("Player Commission HKD", df_f['Player Comm'].sum())

    pk4, pk5 = st.columns(2)
    with pk4:
        draw_box("Player Net Win/Loss HKD", df_f['Player Net HKD'].sum())
    with pk5:
        draw_box("Player Net Win/Loss THB", df_f['Player Net THB'].sum(), "THB")

    # 3. PREPARE DATA
    p_grp = df_f.groupby(
        ['Month_Name', 'Name'],
        as_index=False
    )[
        ['Turnover', 'Gross Player Win/Loss', 'Player Comm', 'Player Net HKD', 'Player Net THB']
    ].sum()

    df_display_p = p_grp.copy()
    
    # --- STEP 1: Add Total Row FIRST (So it finds 'Name' correctly) ---
    df_display_p = add_total_row(df_display_p)

    # --- STEP 2: Make Headers Uppercase (Visual Bold) ---
    df_display_p.columns = df_display_p.columns.str.upper()

    # --- STEP 3: Style Function for Uppercase 'NAME' ---
    def bold_total_upper(row):
        # Now "TOTAL" is definitely in the 'NAME' column
        if row.get('NAME') == 'TOTAL': 
            return ['font-weight: bold; background-color: #f0f0f0'] * len(row)
        return [''] * len(row)

    # 4. DISPLAY TABLE
    st.dataframe(
        df_display_p.style
            .format("{:,.0f}", subset=df_display_p.select_dtypes("number").columns)
            .applymap(highlight_negative)
            .apply(bold_total_upper, axis=1), # This will now work 100%
        use_container_width=True,
        hide_index=True
    )

    # 5. DOWNLOAD BUTTON
    csv_tab2 = convert_df(df_display_p)
    
    st.download_button(
        label="📥 Download Player Report (CSV)",
        data=csv_tab2,
        file_name='Player_Report_Summary.csv',
        mime='text/csv',
    )
with tab3:
    st.subheader("🧾 Player Report Summary (Invoice)")
    
    # --- 1. SYNC WITH SIDEBAR ---
    player_col = 'Name' if 'Name' in df_pool.columns else df_pool.columns[0]
    players_in_month = df_pool[player_col].unique()
    synced_options = sorted([p for p in players_in_month if p in sel_players])
    
    if not synced_options:
        st.warning("⚠️ No players found. Please check your Sidebar filters.")
        st.stop()
        
    inv_p = st.selectbox("Select Player Name:", synced_options)
    
    # --- UI LAYOUT SETUP ---
    invoice_placeholder = st.empty() 
    
    st.divider() 

    # --- 2. SETTINGS ---
    c_set1, c_set2 = st.columns([1, 2])
    with c_set1:
        st.markdown("#### ⚙️ Settings")
        is_thb_input = st.checkbox("☑️ Google Sheet Payments are ALREADY THB", value=True)

    # --- 3. FILTER & CALCULATE ---
    i_d = df_pool[df_pool[player_col] == inv_p].copy()

    # Calculate Totals
    paid_by_raw = i_d['Paid by Player'].sum()
    paid_to_raw = i_d['Paid to Player'].sum()

    if is_thb_input:
        paid_by_thb = paid_by_raw
        paid_to_thb = paid_to_raw
    else:
        paid_by_thb = round(paid_by_raw * hkd_to_thb, 0)
        paid_to_thb = round(paid_to_raw * hkd_to_thb, 0)

    s = {
         'T': i_d['Turnover'].sum(),
         'G': i_d['Gross Player Win/Loss'].sum(),
         'E': i_d['Player Expenses'].sum(), 
         'C': i_d['Player Comm'].sum(),
         'NH': i_d['Player Net HKD'].sum(),  
         'NT': i_d['Player Net THB'].sum(),
         'BY': paid_by_thb,
         'TO': paid_to_thb
    }

    s['Bal'] = round(s['NT'] - paid_to_thb + paid_by_thb, 0)

    # --- 4. RENDER INVOICE ---
    def c(val): return "#d9534f" if val < 0 else "#212529"
    
    with invoice_placeholder.container():
        st.markdown(f"""
        <div class="invoice-container" style="background-color: white; padding: 30px; border: 1px solid #ddd; border-radius: 10px;">
            <div style="display:flex; justify-content:space-between; border-bottom:3px solid #0056b3; padding-bottom:10px;">
                <div style="font-size:24px; font-weight:bold; color: #0056b3;">PLAYER STATEMENT</div>
                <div style="font-weight:bold;">Rate: {hkd_to_thb} THB/HKD</div>
            </div>
            <br>
            <p><b>Player:</b> {inv_p}<br><b>Month(s):</b> {', '.join(sel_months)}</p>
            <table style="width:100%; border-collapse: collapse;">
                <tr style="background:#f8f9fa;"><th style="text-align:left; padding:10px;">Description</th><th style="text-align:right; padding:10px;">HKD</th></tr>
                <tr><td style="padding:10px;">Turnover</td><td style="text-align:right; color:{c(s['T'])};">{s['T']:,.0f}</td></tr>
                <tr><td style="padding:10px;">Gross Win/Loss</td><td style="text-align:right; color:{c(s['G'])};">{s['G']:,.0f}</td></tr>
                <tr><td style="padding:10px;">Player Expenses</td><td style="text-align:right; color:{c(s['E'])};">{s['E']:,.0f}</td></tr>
                <tr><td style="padding:10px;">Commission</td><td style="text-align:right; color:{c(s['C'])};">{s['C']:,.0f}</td></tr>
                <tr style="font-weight:bold; border-top:2px solid #eee;">
                    <td>Net Win/Loss HKD</td><td style="text-align:right; color:{c(s['NH'])};">{s['NH']:,.0f}</td>
                </tr>
            </table>
            <div style="text-align:right; margin-top:20px;">
                <div style="font-size:20px; font-weight:bold; color:{c(s['NT'])};">Net Win/Loss THB: {s['NT']:,.0f} THB</div>
                <div style="color:#666; margin-top:5px;">
                    Paid by Player: <span style="color:{c(s['BY'])}">+{s['BY']:,.0f} THB</span> | 
                    Paid to Player: <span style="color:{c(s['TO'])}">-{s['TO']:,.0f} THB</span>
                </div>
                <hr>
                <div style="font-size:28px; font-weight:bold;">
                    <span style="color: #0056b3;">OUTSTANDING:</span> 
                    <span style="color: {c(s['Bal'])};">{s['Bal']:,.0f} THB</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # PDF Generator
        def pdf_c(pdf, val): 
            if val < 0: pdf.set_text_color(217, 83, 79)
            else: pdf.set_text_color(33, 37, 41)

        def generate_pdf(player, data, rate, months):
            pdf = FPDF()
            pdf.add_page()
            
            # --- TITLE (Blue, Centered) ---
            pdf.set_font("Arial", 'B', 16)
            pdf.set_text_color(0, 86, 179) 
            pdf.cell(0, 10, "PLAYER STATEMENT", ln=True, align='C')
            
            # --- DETAILS (Black, Bold, Left Aligned) ---
            pdf.set_text_color(0, 0, 0) 
            pdf.set_font("Arial", 'B', 12) # BOLD
            
            # Player
            pdf.cell(0, 10, f"Player: {player}", ln=True, align='L')
            
            # Month (Wrapped, Left Aligned)
            month_str = ", ".join(months)
            pdf.multi_cell(0, 10, f"Month: {month_str}", align='L')
            
            # Rate
            pdf.cell(0, 10, f"Rate: {rate} THB/HKD", ln=True, align='L')
            
            pdf.ln(10)
            
            # --- TABLE ---
            rows = [("Turnover", data['T']), ("Gross Win/Loss", data['G']), 
                    ("Player Expenses", data['E']), ("Commission", data['C']), 
                    ("Net Win/Loss HKD", data['NH'])]
            
            pdf.set_fill_color(248, 249, 250)
            pdf.set_font("Arial", '', 12) # Back to normal for table
            pdf.cell(100, 10, "Description", 1, 0, 'L', True)
            pdf.cell(90, 10, "Amount (HKD)", 1, 1, 'R', True)
            
            for desc, val in rows:
                pdf.set_text_color(0, 0, 0)
                pdf.cell(100, 10, desc, 1)
                pdf_c(pdf, val)
                pdf.cell(90, 10, f"{val:,.0f}", 1, 1, 'R')
            
            pdf.ln(10)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(100, 10, "Net Win/Loss THB:")
            pdf_c(pdf, data['NT'])
            pdf.cell(90, 10, f"{data['NT']:,.0f} THB", 0, 1, 'R')
            
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(100, 10, f"Paid by Player: +{data['BY']:,.0f} THB")
            pdf.cell(90, 10, f"Paid to Player: -{data['TO']:,.0f} THB", 0, 1, 'R')
            
            pdf.ln(5)
            pdf.set_draw_color(0, 86, 179)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            
            pdf.set_text_color(0, 86, 179)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(100, 10, "OUTSTANDING:")
            pdf_c(pdf, data['Bal'])
            pdf.cell(90, 10, f"{data['Bal']:,.0f} THB", 0, 1, 'R')
            
            return pdf.output(dest='S').encode('latin-1')

        # Pass 'sel_months' to the function
        pdf_bytes = generate_pdf(inv_p, s, hkd_to_thb, sel_months)
        st.download_button("📥 Download PDF", pdf_bytes, f"Statement_{inv_p}.pdf", "application/pdf")


    # --- 5. PAYMENT HISTORY ---
    st.subheader(f"📋 Payment History ({inv_p})")
    
    payment_mask = (i_d['Paid by Player'].abs() > 0) | (i_d['Paid to Player'].abs() > 0)
    view_df = i_d.loc[payment_mask].copy()
    
    if not view_df.empty:
        if is_thb_input:
            view_df['Paid by (THB)'] = view_df['Paid by Player']
            view_df['Paid to (THB)'] = view_df['Paid to Player']
        else:
            view_df['Paid by (THB)'] = (view_df['Paid by Player'] * hkd_to_thb).round(0)
            view_df['Paid to (THB)'] = (view_df['Paid to Player'] * hkd_to_thb).round(0)
        
        cols = ['Month_Name', 'Paid by Player', 'Paid to Player', 'Paid by (THB)', 'Paid to (THB)']
        final_df = view_df[cols].copy()

        # Add TOTAL Row
        total_row = pd.DataFrame({
            'Month_Name': ['TOTAL'],
            'Paid by Player': [final_df['Paid by Player'].sum()],
            'Paid to Player': [final_df['Paid to Player'].sum()],
            'Paid by (THB)': [final_df['Paid by (THB)'].sum()],
            'Paid to (THB)': [final_df['Paid to (THB)'].sum()]
        })
        
        final_df = pd.concat([final_df, total_row], ignore_index=True)

        format_dict = {
            'Paid by Player': "{:,.0f}",
            'Paid to Player': "{:,.0f}",
            'Paid by (THB)': "{:,.0f}",
            'Paid to (THB)': "{:,.0f}"
        }
        
        st.dataframe(final_df.style.format(format_dict), use_container_width=True, hide_index=True)
    else:
        st.info("No individual payment records found.")