# E-commerce Analytics Dashboard

## Overview

The E-commerce Analytics Dashboard is an interactive web application built with Streamlit that provides comprehensive visualization and analysis of e-commerce sales data. The dashboard displays key business metrics, trends, and insights from the data processing pipeline.

## Features

### 📊 Interactive Visualizations

- **Monthly Revenue Trends**: Line, bar, and area charts showing revenue patterns over time
- **Top Products Analysis**: Bar charts displaying top-performing products by revenue and units sold
- **Regional Performance**: Multi-panel visualization showing sales metrics across different regions
- **Category Discount Heatmap**: Interactive heatmap showing discount effectiveness by product category
- **Anomaly Detection**: Dedicated section for reviewing suspicious transactions

### 🔍 Interactive Filtering

- **Date Range Filtering**: Filter data by specific months
- **Regional Filtering**: Focus on specific geographic regions
- **Category Filtering**: Analyze specific product categories
- **Dynamic Updates**: All visualizations update automatically based on applied filters

### 📈 Key Performance Indicators

- Total Revenue with growth indicators
- Total Orders with trend analysis
- Average Order Value tracking
- Average Discount rates

### 🚀 Performance Features

- **Data Caching**: 5-minute cache for improved performance
- **Responsive Design**: Mobile-friendly interface
- **Real-time Updates**: Refresh data on demand
- **Export Capabilities**: Download filtered data and dashboard exports

## Requirements

The dashboard requires the following dependencies (included in `requirements.txt`):

- `streamlit>=1.25.0`
- `plotly>=5.15.0`
- `pandas>=1.5.0`
- `numpy>=1.21.0`

## Usage

### Running the Dashboard

1. **Using the run script**:
   ```bash
   python run_dashboard.py
   ```

2. **Using Streamlit directly**:
   ```bash
   streamlit run src/dashboard/dashboard_app.py
   ```

3. **Testing the dashboard**:
   ```bash
   python test_dashboard.py
   ```

### Dashboard Controls

#### Sidebar Controls
- **🔄 Refresh Data**: Manually refresh the data cache
- **🔍 Filters**: Enable/disable various data filters
- **📊 Performance**: Monitor data freshness and availability

#### Main Interface
- **Chart Type Selection**: Choose between different visualization types
- **Product Count Slider**: Adjust the number of products displayed
- **Color Scale Selection**: Customize heatmap color schemes
- **Detail Toggles**: Show/hide detailed data tables

### Data Sources

The dashboard automatically loads data from the following analytical tables:

- `monthly_sales_summary.csv` - Monthly aggregated sales metrics
- `top_products.csv` - Top-performing products analysis
- `region_wise_performance.csv` - Regional sales performance
- `category_discount_map.csv` - Category discount analysis
- `anomaly_records.csv` - Detected anomalous transactions

If actual data files are not available, the dashboard will generate sample data for development and testing purposes.

## Architecture

### Components

1. **DashboardApp** (`dashboard_app.py`)
   - Main Streamlit application
   - Chart creation and visualization logic
   - Interactive controls and filtering
   - User interface management

2. **DataProvider** (`data_provider.py`)
   - Data loading and caching
   - Metrics calculation
   - Data freshness monitoring
   - Export functionality

### Data Flow

```
Data Files → DataProvider → Dashboard App → Streamlit Interface
     ↓            ↓              ↓              ↓
   CSV/Parquet   Caching    Visualization   User Interaction
```

## Customization

### Adding New Visualizations

To add a new chart type:

1. Create a new method in `DashboardApp` class:
   ```python
   def create_new_chart(self, data: pd.DataFrame) -> go.Figure:
       # Chart creation logic
       pass
   ```

2. Add the chart to the main dashboard in the `run()` method:
   ```python
   if 'new_data' in analytical_tables:
       new_chart = self.create_new_chart(analytical_tables['new_data'])
       st.plotly_chart(new_chart, use_container_width=True)
   ```

### Adding New Filters

To add a new filter:

1. Add filter control in the sidebar section of `run()` method
2. Update the `_apply_filters()` method to handle the new filter
3. Update the `_calculate_filtered_metrics()` method if needed

### Customizing Styling

The dashboard uses custom CSS defined in the `setup_page_config()` method. Modify the CSS styles to change the appearance:

```python
st.markdown("""
<style>
.custom-style {
    /* Your custom styles */
}
</style>
""", unsafe_allow_html=True)
```

## Performance Optimization

### Caching Strategy

- Data is cached for 5 minutes to reduce file I/O
- Cache validation checks prevent stale data display
- Manual refresh option for immediate updates

### Memory Management

- Efficient DataFrame operations
- Minimal data copying during filtering
- Garbage collection friendly design

### Loading Times

- Lazy loading of visualizations
- Streamlit's built-in caching mechanisms
- Optimized data structures

## Troubleshooting

### Common Issues

1. **"No data available" messages**
   - Ensure analytical tables are generated by running the ETL pipeline
   - Check that files exist in the `data/output` directory
   - Verify file permissions and accessibility

2. **Slow loading times**
   - Check data file sizes
   - Clear browser cache
   - Restart the Streamlit server

3. **Chart rendering issues**
   - Update Plotly to the latest version
   - Check browser compatibility
   - Verify data format consistency

### Debug Mode

Run the test script to verify component functionality:
```bash
python test_dashboard.py
```

### Logging

The dashboard uses Python's logging module. To enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- Real-time data streaming
- Advanced filtering options
- Custom dashboard layouts
- User authentication and personalization
- Mobile app version
- API integration for external data sources

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test script to verify functionality
3. Review the logs for error messages
4. Ensure all dependencies are properly installed