from flask import Flask, render_template, jsonify, request
from datetime import datetime

app = Flask(__name__)


# ================================================================
#  MOCK DATA — replaced by real pipeline in production
# ================================================================

def get_kpis():
    return {
        'total_revenue': 87410,
        'revenue_change': 10.3,
        'waste_saved': 12430,
        'waste_saved_change': 23.5,
        'spoilage_rate': 4.2,
        'spoilage_change': -2.1,
        'active_discounts': 8,
        'discounts_change': 3,
    }


def get_ppo_suggestions():
    return [
        {
            'id': 1, 'product': 'Amul Toned Milk 500ml', 'category': 'Dairy',
            'current_price': 30, 'discount_pct': 20, 'new_price': 24,
            'confidence': 92, 'days_to_expiry': 1, 'stock': 45,
            'predicted_demand': 30, 'status': 'pending',
            'reason': 'Expiring tomorrow — stock exceeds predicted demand by 50%',
        },
        {
            'id': 2, 'product': 'Mother Dairy Curd 400g', 'category': 'Dairy',
            'current_price': 40, 'discount_pct': 15, 'new_price': 34,
            'confidence': 87, 'days_to_expiry': 2, 'stock': 32,
            'predicted_demand': 25, 'status': 'pending',
            'reason': 'Weather: rain expected — footfall likely to drop 20%',
        },
        {
            'id': 3, 'product': 'Britannia Bread 400g', 'category': 'Bakery',
            'current_price': 45, 'discount_pct': 25, 'new_price': 33.75,
            'confidence': 95, 'days_to_expiry': 1, 'stock': 20,
            'predicted_demand': 12, 'status': 'pending',
            'reason': 'Critical: expires tomorrow, surplus of 8 units projected',
        },
        {
            'id': 4, 'product': 'Fresh Paneer 200g', 'category': 'Dairy',
            'current_price': 80, 'discount_pct': 30, 'new_price': 56,
            'confidence': 88, 'days_to_expiry': 1, 'stock': 15,
            'predicted_demand': 8, 'status': 'pending',
            'reason': 'Premium item — high waste cost if unsold (₹80/unit)',
        },
        {
            'id': 5, 'product': 'Tomatoes 1kg', 'category': 'Fruits & Vegetables',
            'current_price': 40, 'discount_pct': 10, 'new_price': 36,
            'confidence': 78, 'days_to_expiry': 3, 'stock': 60,
            'predicted_demand': 45, 'status': 'pending',
            'reason': 'Seasonal oversupply — competitor pricing ₹35/kg',
        },
        {
            'id': 6, 'product': 'Bananas 1 dozen', 'category': 'Fruits & Vegetables',
            'current_price': 50, 'discount_pct': 15, 'new_price': 42.50,
            'confidence': 82, 'days_to_expiry': 2, 'stock': 38,
            'predicted_demand': 28, 'status': 'pending',
            'reason': 'Ripening fast — weekday demand historically 30% lower',
        },
        {
            'id': 7, 'product': 'Chicken Seekh Kebab 250g', 'category': 'Fast Food',
            'current_price': 120, 'discount_pct': 20, 'new_price': 96,
            'confidence': 90, 'days_to_expiry': 1, 'stock': 10,
            'predicted_demand': 6, 'status': 'pending',
            'reason': 'Frozen item expiring — high value recovery opportunity',
        },
        {
            'id': 8, 'product': 'Nandini Buttermilk 200ml', 'category': 'Dairy',
            'current_price': 15, 'discount_pct': 20, 'new_price': 12,
            'confidence': 85, 'days_to_expiry': 1, 'stock': 55,
            'predicted_demand': 35, 'status': 'pending',
            'reason': 'High stock, low shelf life — clear before tomorrow',
        },
    ]


def get_top_products():
    return [
        {'name': 'Amul Butter 100g', 'category': 'Dairy', 'revenue': 4960, 'units': 124, 'trend': 12.5},
        {'name': 'Tomatoes 1kg', 'category': 'F&V', 'revenue': 3800, 'units': 95, 'trend': 8.2},
        {'name': 'Mother Dairy Milk 1L', 'category': 'Dairy', 'revenue': 3550, 'units': 71, 'trend': -2.1},
        {'name': 'Onions 1kg', 'category': 'F&V', 'revenue': 3200, 'units': 80, 'trend': 15.3},
        {'name': 'Britannia Bread 400g', 'category': 'Bakery', 'revenue': 2700, 'units': 60, 'trend': 5.7},
    ]


def get_expiry_items():
    return [
        {'id': 1, 'product': 'Fresh Paneer 200g', 'category': 'Dairy', 'stock': 15, 'days_left': 0, 'urgency': 'critical', 'value_at_risk': 1200, 'batch': 'B-4521'},
        {'id': 2, 'product': 'Britannia Bread 400g', 'category': 'Bakery', 'stock': 20, 'days_left': 1, 'urgency': 'critical', 'value_at_risk': 900, 'batch': 'B-4518'},
        {'id': 3, 'product': 'Amul Toned Milk 500ml', 'category': 'Dairy', 'stock': 45, 'days_left': 1, 'urgency': 'critical', 'value_at_risk': 1350, 'batch': 'B-4523'},
        {'id': 4, 'product': 'Chicken Seekh Kebab 250g', 'category': 'Fast Food', 'stock': 10, 'days_left': 1, 'urgency': 'critical', 'value_at_risk': 1200, 'batch': 'B-4515'},
        {'id': 5, 'product': 'Mother Dairy Curd 400g', 'category': 'Dairy', 'stock': 32, 'days_left': 2, 'urgency': 'warning', 'value_at_risk': 1280, 'batch': 'B-4520'},
        {'id': 6, 'product': 'Bananas 1 dozen', 'category': 'F&V', 'stock': 38, 'days_left': 2, 'urgency': 'warning', 'value_at_risk': 1900, 'batch': 'B-4519'},
        {'id': 7, 'product': 'Nandini Buttermilk 200ml', 'category': 'Dairy', 'stock': 55, 'days_left': 1, 'urgency': 'critical', 'value_at_risk': 825, 'batch': 'B-4524'},
        {'id': 8, 'product': 'Tomatoes 1kg', 'category': 'F&V', 'stock': 60, 'days_left': 3, 'urgency': 'normal', 'value_at_risk': 2400, 'batch': 'B-4522'},
        {'id': 9, 'product': 'Apples 1kg', 'category': 'F&V', 'stock': 25, 'days_left': 5, 'urgency': 'normal', 'value_at_risk': 3750, 'batch': 'B-4516'},
        {'id': 10, 'product': 'Chole Bhature Pack', 'category': 'Fast Food', 'stock': 8, 'days_left': 0, 'urgency': 'critical', 'value_at_risk': 640, 'batch': 'B-4525'},
    ]


def get_sales_data():
    return {
        'daily_labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'daily_revenue': [12400, 15600, 13200, 17800, 14500, 21000, 18900],
        'daily_units': [310, 390, 330, 445, 362, 525, 472],
        'weekly_labels': ['W1', 'W2', 'W3', 'W4'],
        'weekly_revenue': [82500, 91200, 87410, 95300],
        'monthly_labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
        'monthly_revenue': [78000, 82000, 75000, 89000, 92000, 85000, 97000, 87410, 91000],
        'category_labels': ['Dairy', 'Fruits & Veg', 'Fast Food', 'Bakery'],
        'category_revenue': [34500, 28200, 15800, 8910],
        'hourly_labels': ['6AM', '8AM', '10AM', '12PM', '2PM', '4PM', '6PM', '8PM', '10PM'],
        'hourly_footfall': [20, 65, 120, 180, 95, 140, 200, 150, 45],
    }


def get_forecast_data():
    return {
        'products': ['Amul Milk 500ml', 'Tomatoes 1kg', 'Britannia Bread', 'Mother Dairy Curd', 'Bananas 1 dozen'],
        'labels': ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
        'predicted': [45, 38, 52, 41, 55, 67, 48],
        'actual': [42, 40, 49, 44, 52, 63, 50],
        'accuracy': {'rmse': 2.8, 'mae': 2.3, 'mape': 5.7},
        'features': {
            'labels': ['Days to Expiry', 'Temperature', 'Day of Week', 'Is Holiday',
                       'Price', 'Rolling Avg 7d', 'Humidity', 'Precipitation', 'Month', 'Is Weekend'],
            'importance': [0.28, 0.18, 0.14, 0.11, 0.09, 0.07, 0.05, 0.04, 0.02, 0.02],
        },
    }


def get_alerts():
    return [
        {'type': 'critical', 'message': '6 products expiring today — immediate action needed', 'icon': 'alert-triangle'},
        {'type': 'info', 'message': 'PPO model suggests 8 new discounts for review', 'icon': 'cpu'},
        {'type': 'success', 'message': "Yesterday's discounts saved ₹3,200 in potential waste", 'icon': 'check-circle'},
    ]


def get_discount_history():
    return [
        {'date': '29 Sep 2024', 'product': 'Amul Milk 500ml', 'discount': 15, 'action': 'accepted', 'result': 'Sold 38/45 units', 'revenue_impact': '+₹912'},
        {'date': '29 Sep 2024', 'product': 'Paneer 200g', 'discount': 25, 'action': 'accepted', 'result': 'Sold 12/15 units', 'revenue_impact': '+₹672'},
        {'date': '28 Sep 2024', 'product': 'Bread 400g', 'discount': 20, 'action': 'rejected', 'result': '8 units wasted', 'revenue_impact': '-₹360'},
        {'date': '28 Sep 2024', 'product': 'Curd 400g', 'discount': 10, 'action': 'accepted', 'result': 'Sold 28/30 units', 'revenue_impact': '+₹1,008'},
        {'date': '27 Sep 2024', 'product': 'Tomatoes 1kg', 'discount': 15, 'action': 'accepted', 'result': 'Sold 50/55 units', 'revenue_impact': '+₹1,700'},
    ]


# ================================================================
#  CONTEXT PROCESSOR — injects globals into every template
# ================================================================

@app.context_processor
def inject_globals():
    suggestions = get_ppo_suggestions()
    expiry = get_expiry_items()
    return {
        'pending_suggestions': len([s for s in suggestions if s['status'] == 'pending']),
        'critical_count': sum(1 for e in expiry if e['urgency'] == 'critical'),
        'current_time': datetime.now().strftime('%I:%M %p'),
        'current_date': datetime.now().strftime('%A, %d %B %Y'),
    }


# ================================================================
#  ROUTES
# ================================================================

@app.route('/')
def overview():
    return render_template(
        'overview.html',
        active_page='overview',
        kpis=get_kpis(),
        alerts=get_alerts(),
        top_products=get_top_products(),
        sales=get_sales_data(),
    )


@app.route('/sales')
def sales():
    return render_template(
        'sales.html',
        active_page='sales',
        sales=get_sales_data(),
        top_products=get_top_products(),
    )


@app.route('/ppo')
def ppo():
    suggestions = get_ppo_suggestions()
    accepted = [s for s in suggestions if s['status'] == 'accepted']
    return render_template(
        'ppo.html',
        active_page='ppo',
        suggestions=suggestions,
        history=get_discount_history(),
        total_potential_savings=sum((s['current_price'] - s['new_price']) * s['stock'] for s in suggestions),
    )


@app.route('/forecast')
def forecast():
    return render_template(
        'forecast.html',
        active_page='forecast',
        forecast=get_forecast_data(),
    )


@app.route('/expiry')
def expiry():
    items = get_expiry_items()
    return render_template(
        'expiry.html',
        active_page='expiry',
        items=items,
        total_at_risk=sum(e['value_at_risk'] for e in items),
        critical_count_page=sum(1 for e in items if e['urgency'] == 'critical'),
        warning_count=sum(1 for e in items if e['urgency'] == 'warning'),
    )


@app.route('/settings')
def settings():
    return render_template('settings.html', active_page='settings')


# ================================================================
#  API ENDPOINTS (for AJAX interactions)
# ================================================================

@app.route('/api/suggestion/<int:sid>/<action>', methods=['POST'])
def update_suggestion(sid, action):
    """Accept, reject, or implement a PPO suggestion."""
    return jsonify({'success': True, 'id': sid, 'action': action, 'timestamp': datetime.now().isoformat()})


@app.route('/api/bulk-action', methods=['POST'])
def bulk_action():
    """Bulk accept/reject/implement all suggestions."""
    data = request.get_json()
    return jsonify({'success': True, 'action': data.get('action'), 'count': data.get('count', 0)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
