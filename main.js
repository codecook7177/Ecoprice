/* ================================================================
   ECOPRICE — Main JavaScript
   Scroll animations · Counter animation · Sidebar · Interactions
   ================================================================ */

document.addEventListener('DOMContentLoaded', () => {

    // ---- Scroll Reveal (Intersection Observer) ----
    const revealElements = document.querySelectorAll(
        '.scroll-reveal, .slide-left, .slide-right, .scale-in'
    );

    const revealObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                    revealObserver.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.08, rootMargin: '0px 0px -40px 0px' }
    );

    revealElements.forEach((el) => revealObserver.observe(el));


    // ---- Animated Counters ----
    const counters = document.querySelectorAll('[data-count]');

    const counterObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    animateCounter(entry.target);
                    counterObserver.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.5 }
    );

    counters.forEach((el) => counterObserver.observe(el));

    function animateCounter(el) {
        const target = parseFloat(el.dataset.count);
        const prefix = el.dataset.prefix || '';
        const suffix = el.dataset.suffix || '';
        const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 0;
        const duration = 1200;
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = eased * target;

            el.textContent = prefix + formatNumber(current, decimals) + suffix;

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                el.textContent = prefix + formatNumber(target, decimals) + suffix;
            }
        }

        requestAnimationFrame(update);
    }

    function formatNumber(num, decimals) {
        if (decimals > 0) {
            return num.toFixed(decimals);
        }
        return Math.round(num).toLocaleString('en-IN');
    }


    // ---- Sidebar Toggle (Mobile) ----
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle');

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', (e) => {
            if (
                window.innerWidth <= 768 &&
                sidebar.classList.contains('open') &&
                !sidebar.contains(e.target) &&
                !toggleBtn.contains(e.target)
            ) {
                sidebar.classList.remove('open');
            }
        });
    }


    // ---- Filter Buttons ----
    const filterGroups = document.querySelectorAll('.filter-bar');
    filterGroups.forEach((group) => {
        const buttons = group.querySelectorAll('.filter-btn');
        buttons.forEach((btn) => {
            btn.addEventListener('click', () => {
                buttons.forEach((b) => b.classList.remove('active'));
                btn.classList.add('active');
                // Trigger filter logic if needed
                const filterValue = btn.dataset.filter;
                if (filterValue) {
                    filterTable(group, filterValue);
                }
            });
        });
    });

    function filterTable(group, value) {
        const tableId = group.dataset.tableTarget;
        if (!tableId) return;
        const table = document.getElementById(tableId);
        if (!table) return;

        const rows = table.querySelectorAll('tbody tr');
        rows.forEach((row) => {
            if (value === 'all') {
                row.style.display = '';
                row.style.opacity = '0';
                row.style.transform = 'translateY(10px)';
                requestAnimationFrame(() => {
                    row.style.transition = 'opacity 0.3s, transform 0.3s';
                    row.style.opacity = '1';
                    row.style.transform = 'translateY(0)';
                });
            } else {
                const category = row.dataset.category || '';
                if (category.toLowerCase().includes(value.toLowerCase())) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            }
        });
    }


    // ---- PPO Suggestion Actions ----
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-action]');
        if (!btn) return;

        const action = btn.dataset.action;
        const row = btn.closest('tr') || btn.closest('.suggestion-row');
        const id = btn.dataset.id;

        if (action === 'accept') {
            handleAccept(btn, row, id);
        } else if (action === 'reject') {
            handleReject(btn, row, id);
        } else if (action === 'implement') {
            handleImplement(btn, row, id);
        } else if (action === 'undo') {
            handleUndo(btn, row, id);
        }
    });

    function handleAccept(btn, row, id) {
        if (!row) return;
        // Animate the row
        row.style.transition = 'background 0.3s ease';
        row.style.background = 'rgba(16, 185, 129, 0.06)';

        // Update badge
        const badge = row.querySelector('.status-badge');
        if (badge) {
            badge.className = 'badge badge-accepted status-badge';
            badge.textContent = 'Accepted';
        }

        // Show implement button, hide accept/reject
        const actionBtns = row.querySelector('.action-btns');
        if (actionBtns) {
            actionBtns.innerHTML = `
                <button class="btn btn-implement btn-sm" data-action="implement" data-id="${id}">
                    <i data-lucide="check-circle"></i> Implement
                </button>
                <button class="btn btn-outline btn-sm" data-action="undo" data-id="${id}">Undo</button>
            `;
            lucide.createIcons();
        }

        showToast('Discount accepted', 'success');
        callAPI(id, 'accept');
    }

    function handleReject(btn, row, id) {
        if (!row) return;
        row.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
        row.style.opacity = '0.4';

        const badge = row.querySelector('.status-badge');
        if (badge) {
            badge.className = 'badge badge-rejected status-badge';
            badge.textContent = 'Rejected';
        }

        const actionBtns = row.querySelector('.action-btns');
        if (actionBtns) {
            actionBtns.innerHTML = `
                <button class="btn btn-outline btn-sm" data-action="undo" data-id="${id}">Undo</button>
            `;
        }

        showToast('Discount rejected', 'info');
        callAPI(id, 'reject');
    }

    function handleImplement(btn, row, id) {
        if (!row) return;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Implementing...';

        setTimeout(() => {
            row.style.background = 'rgba(16, 185, 129, 0.1)';

            const badge = row.querySelector('.status-badge');
            if (badge) {
                badge.className = 'badge badge-implemented status-badge';
                badge.textContent = 'Active ✓';
            }

            const actionBtns = row.querySelector('.action-btns');
            if (actionBtns) {
                actionBtns.innerHTML = `
                    <span class="badge badge-active">Live</span>
                `;
            }

            // Extract product info and add to Live Prices section
            const productName = row.querySelector('.product-name')?.textContent?.trim() || 'Product';
            const productCat = row.querySelector('.product-category')?.textContent?.trim() || '';
            const oldPrice = row.querySelector('.price-strike')?.textContent?.trim() || '';
            const newPrice = row.querySelector('.price-new')?.textContent?.trim() || '';
            const discountEl = row.querySelector('.text-emerald');
            const discount = discountEl ? discountEl.textContent.trim().replace('-','').replace('%','') : '';

            if (window.addToLivePrices) {
                window.addToLivePrices(productName, productCat, oldPrice, newPrice, discount);
            }

            showToast('Discount implemented — price updated in system', 'success');
            callAPI(id, 'implement');
        }, 800);
    }

    function handleUndo(btn, row, id) {
        if (!row) return;
        row.style.transition = 'all 0.3s ease';
        row.style.background = '';
        row.style.opacity = '1';

        const badge = row.querySelector('.status-badge');
        if (badge) {
            badge.className = 'badge badge-pending status-badge';
            badge.textContent = 'Pending';
        }

        const actionBtns = row.querySelector('.action-btns');
        if (actionBtns) {
            actionBtns.innerHTML = `
                <button class="btn btn-accept btn-sm" data-action="accept" data-id="${id}">Accept</button>
                <button class="btn btn-reject btn-sm" data-action="reject" data-id="${id}">Reject</button>
            `;
        }

        showToast('Action reverted', 'info');
    }


    // ---- Bulk Actions ----
    document.addEventListener('click', (e) => {
        const bulkBtn = e.target.closest('[data-bulk]');
        if (!bulkBtn) return;

        const action = bulkBtn.dataset.bulk;
        const rows = document.querySelectorAll('#suggestions-table tbody tr');

        rows.forEach((row, i) => {
            setTimeout(() => {
                const acceptBtn = row.querySelector('[data-action="accept"]');
                const rejectBtn = row.querySelector('[data-action="reject"]');

                if (action === 'accept-all' && acceptBtn) {
                    acceptBtn.click();
                } else if (action === 'reject-all' && rejectBtn) {
                    rejectBtn.click();
                }
            }, i * 100); // Stagger for visual effect
        });
    });


    // ---- Toast Notifications ----
    function showToast(message, type = 'info') {
        const existing = document.querySelector('.toast');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `<span>${message}</span>`;

        Object.assign(toast.style, {
            position: 'fixed',
            bottom: '24px',
            right: '24px',
            padding: '0.75rem 1.25rem',
            borderRadius: '10px',
            background: type === 'success' ? '#059669' : type === 'info' ? '#3b82f6' : '#ef4444',
            color: 'white',
            fontSize: '0.85rem',
            fontWeight: '600',
            fontFamily: 'Inter, sans-serif',
            boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
            zIndex: '9999',
            opacity: '0',
            transform: 'translateY(16px)',
            transition: 'opacity 0.3s, transform 0.3s',
        });

        document.body.appendChild(toast);

        requestAnimationFrame(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateY(0)';
        });

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(16px)';
            setTimeout(() => toast.remove(), 300);
        }, 2800);
    }

    // Make it globally accessible
    window.showToast = showToast;


    // ---- API Calls ----
    function callAPI(id, action) {
        fetch(`/api/suggestion/${id}/${action}`, { method: 'POST' }).catch(() => {});
    }


    // ---- Confidence Bar Animation ----
    const confidenceBars = document.querySelectorAll('.confidence-bar .fill');
    const barObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    const width = entry.target.dataset.width;
                    entry.target.style.width = width + '%';
                    barObserver.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.5 }
    );
    confidenceBars.forEach((bar) => {
        bar.style.width = '0%';
        barObserver.observe(bar);
    });


    // ---- Print Price Tags ----
    window.printPriceTags = function () {
        const liveRows = document.querySelectorAll('#live-prices-tbody tr');
        
        if (!liveRows || liveRows.length === 0) {
            showToast('No active discounts to print. Implement a discount first.', 'warning');
            return;
        }

        let tagsHTML = '';
        liveRows.forEach(row => {
            const product = row.querySelector('.product-name')?.textContent || 'Product';
            const oldPrice = row.querySelector('.price-strike')?.textContent || '';
            const newPrice = row.querySelector('.price-new')?.textContent || '';
            const discountBadge = row.querySelector('.badge-accepted');
            const discount = discountBadge ? discountBadge.textContent.replace('-', '').replace('%', '') : '';

            tagsHTML += `
                <div class="tag">
                    <div class="tag-title">EcoPrice Deal</div>
                    <div class="tag-product">${product}</div>
                    <div class="tag-prices">
                        <span class="tag-old">${oldPrice}</span>
                        <span class="tag-new">${newPrice}</span>
                    </div>
                    <div class="tag-discount">${discount}% OFF</div>
                    <div class="tag-footer">Valid Today Only</div>
                </div>
            `;
        });

        const printWindow = window.open('', '_blank');
        printWindow.document.write(`
            <html>
            <head>
                <title>EcoPrice — Price Tags</title>
                <style>
                    body { font-family: 'Inter', Arial, sans-serif; padding: 20px; text-align: center; }
                    .tag { border: 2px dashed #059669; border-radius: 12px; padding: 20px; text-align: center; margin: 10px; display: inline-block; width: 250px; page-break-inside: avoid; }
                    .tag-title { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; color: #059669; }
                    .tag-product { font-size: 16px; font-weight: 700; margin: 8px 0; color: #333; }
                    .tag-prices { font-size: 20px; margin: 10px 0; }
                    .tag-old { text-decoration: line-through; color: #999; margin-right: 10px; }
                    .tag-new { color: #059669; font-weight: 800; font-size: 24px; }
                    .tag-discount { background: #059669; color: white; padding: 4px 10px; border-radius: 20px; font-size: 14px; font-weight: 700; display: inline-block; }
                    .tag-footer { font-size: 10px; color: #999; margin-top: 12px; }
                </style>
            </head>
            <body>
                <h2 style="font-family: 'Inter', sans-serif; color: #333; margin-bottom: 20px;">EcoPrice — Shelf Labels</h2>
                ${tagsHTML}
            </body>
            </html>
        `);
        printWindow.document.close();
        
        // Brief timeout to ensure styles load before print dialog opens
        setTimeout(() => {
            printWindow.print();
        }, 250);
    };


    // ---- Dark Mode Toggle ----
    const themeToggle = document.getElementById('theme-toggle');
    const iconDark = document.getElementById('theme-icon-dark');
    const iconLight = document.getElementById('theme-icon-light');

    // Restore saved theme
    if (localStorage.getItem('ecoprice-theme') === 'dark') {
        document.body.classList.add('dark-mode');
        if (iconDark) iconDark.style.display = 'none';
        if (iconLight) iconLight.style.display = 'block';
    }

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            localStorage.setItem('ecoprice-theme', isDark ? 'dark' : 'light');

            if (iconDark && iconLight) {
                iconDark.style.display = isDark ? 'none' : 'block';
                iconLight.style.display = isDark ? 'block' : 'none';
            }

            // Smooth transition for bg
            document.body.style.transition = 'background 0.5s ease, color 0.3s ease';
        });
    }


    // ---- Live Prices Tracking (PPO page) ----
    window.livePrices = window.livePrices || [];

    window.addToLivePrices = function(product, category, oldPrice, newPrice, discount) {
        const section = document.getElementById('live-prices-section');
        const tbody = document.getElementById('live-prices-tbody');
        if (!section || !tbody) return;

        section.classList.add('visible');

        const now = new Date();
        const time = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });

        const row = document.createElement('tr');
        row.style.opacity = '0';
        row.style.transform = 'translateX(-10px)';
        row.innerHTML = `
            <td>
                <div class="product-cell">
                    <span class="product-name">${product}</span>
                    <span class="product-category">${category}</span>
                </div>
            </td>
            <td><span class="price-strike">${oldPrice}</span></td>
            <td><span class="price-new fw-700">${newPrice}</span></td>
            <td><span class="badge badge-accepted">-${discount}%</span></td>
            <td><span class="badge badge-active">Live</span></td>
            <td class="fs-sm text-gray">${time}</td>
        `;
        tbody.prepend(row);

        requestAnimationFrame(() => {
            row.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
            row.style.opacity = '1';
            row.style.transform = 'translateX(0)';
        });

        // Update count
        const countEl = document.getElementById('live-prices-count');
        if (countEl) {
            const count = tbody.querySelectorAll('tr').length;
            countEl.textContent = count;
        }
    };
});

