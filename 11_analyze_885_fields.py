from pim_gan.plots.table_s7 import export_table_s7_summary

def main():
    df = export_table_s7_summary()
    print("=== Table S7: 885 °C Field-level Statistical Analysis ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
