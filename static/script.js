/**
 * AI Fraud Sentinel - Client Interactions, Pickers & Interactive Clock Widget
 */
console.log("AI Fraud Sentinel JavaScript loaded");
document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // 1. Elements
    // -------------------------------------------------------------
    const dateInput = document.getElementById('transaction_date');
    const timeInput = document.getElementById('transaction_time');
    const hiddenDtInput = document.getElementById('transaction_datetime');
    const dobInput = document.getElementById('date_of_birth');

    const datePickerBtn = document.getElementById('date-picker-btn');
    const timePickerBtn = document.getElementById('time-picker-btn');
    const dobPickerBtn = document.getElementById('dob-picker-btn');

    const clockDropdown = document.getElementById('clock-dropdown');
    const clockHourVal = document.getElementById('clock-hour-val');
    const clockMinVal = document.getElementById('clock-min-val');
    const amBtn = document.getElementById('ampm-am');
    const pmBtn = document.getElementById('ampm-pm');
    const clockNowBtn = document.getElementById('clock-now-btn');
    const clockDoneBtn = document.getElementById('clock-done-btn');

    const transactionForm = document.getElementById('transaction-form');
    const resultSection = document.getElementById('result');
    const errorElement = document.getElementById('error');
    const submitButton = document.getElementById('submit-btn');

    // -------------------------------------------------------------
    // 2. Initialize Default Values
    // -------------------------------------------------------------
    const now = new Date();
    const pad = (num) => String(num).padStart(2, '0');

    if (dateInput && !dateInput.value) {
        dateInput.value =
            `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
    }

    if (timeInput && !timeInput.value) {
        timeInput.value = `${pad(now.getHours())}:${pad(now.getMinutes())}`;
    }

    function syncDateTime() {
        if (!hiddenDtInput) return;

        const date = dateInput?.value || '';
        const time = timeInput?.value || '00:00';

        if (date) {
            hiddenDtInput.value = `${date}T${time}`;
        }
    }

    syncDateTime();

    dateInput?.addEventListener('change', syncDateTime);
    timeInput?.addEventListener('change', syncDateTime);

    // Prevent selecting a date of birth in the future.
    if (dobInput) {
        const todayStr =
            `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;

        dobInput.setAttribute('max', todayStr);
    }

    // -------------------------------------------------------------
    // 3. Calendar Pickers
    // -------------------------------------------------------------
    function triggerDatePicker(input) {
        if (!input) return;

        try {
            if (typeof input.showPicker === 'function') {
                input.showPicker();
            } else {
                input.focus();
            }
        } catch (error) {
            input.focus();
        }
    }

    datePickerBtn?.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        triggerDatePicker(dateInput);
    });

    dateInput?.addEventListener('click', () => {
        triggerDatePicker(dateInput);
    });

    dobPickerBtn?.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        triggerDatePicker(dobInput);
    });

    dobInput?.addEventListener('click', () => {
        triggerDatePicker(dobInput);
    });

    // -------------------------------------------------------------
    // 4. Interactive Clock Widget
    // -------------------------------------------------------------
    let currentHour12 = 12;
    let currentMin = 0;
    let currentAmPm = 'PM';

    function loadTimeIntoClock() {
        if (!timeInput) return;

        const [hourString, minuteString] =
            (timeInput.value || '12:00').split(':');

        let hour = parseInt(hourString, 10);
        let minute = parseInt(minuteString, 10);

        if (Number.isNaN(hour)) hour = 12;
        if (Number.isNaN(minute)) minute = 0;

        currentAmPm = hour >= 12 ? 'PM' : 'AM';
        currentHour12 = hour % 12;

        if (currentHour12 === 0) {
            currentHour12 = 12;
        }

        currentMin = minute;
        renderClockDisplay();
    }

    function renderClockDisplay() {
        if (clockHourVal) {
            clockHourVal.textContent = pad(currentHour12);
        }

        if (clockMinVal) {
            clockMinVal.textContent = pad(currentMin);
        }

        if (amBtn && pmBtn) {
            amBtn.classList.toggle('active', currentAmPm === 'AM');
            pmBtn.classList.toggle('active', currentAmPm === 'PM');
        }
    }

    function commitClockToInput() {
        let hour24 = currentHour12 % 12;

        if (currentAmPm === 'PM') {
            hour24 += 12;
        }

        if (timeInput) {
            timeInput.value = `${pad(hour24)}:${pad(currentMin)}`;
            timeInput.dispatchEvent(
                new Event('change', { bubbles: true })
            );
        }

        syncDateTime();
    }

    function openClock() {
        if (!clockDropdown) return;

        loadTimeIntoClock();
        clockDropdown.removeAttribute('hidden');
    }

    function closeClock() {
        if (!clockDropdown) return;

        clockDropdown.setAttribute('hidden', '');
    }

    function toggleClock(event) {
        event?.preventDefault();
        event?.stopPropagation();

        if (!clockDropdown) return;

        if (clockDropdown.hasAttribute('hidden')) {
            openClock();
        } else {
            closeClock();
        }
    }

    timePickerBtn?.addEventListener('click', toggleClock);

    timeInput?.addEventListener('click', (event) => {
        event.stopPropagation();
        openClock();
    });

    // Hour controls
    document.getElementById('hour-up')?.addEventListener('click', (event) => {
        event.stopPropagation();

        currentHour12 = currentHour12 >= 12 ? 1 : currentHour12 + 1;

        renderClockDisplay();
        commitClockToInput();
    });

    document.getElementById('hour-down')?.addEventListener('click', (event) => {
        event.stopPropagation();

        currentHour12 = currentHour12 <= 1 ? 12 : currentHour12 - 1;

        renderClockDisplay();
        commitClockToInput();
    });

    // Minute controls
    document.getElementById('min-up')?.addEventListener('click', (event) => {
        event.stopPropagation();

        currentMin = (currentMin + 1) % 60;

        renderClockDisplay();
        commitClockToInput();
    });

    document.getElementById('min-down')?.addEventListener('click', (event) => {
        event.stopPropagation();

        currentMin = (currentMin - 1 + 60) % 60;

        renderClockDisplay();
        commitClockToInput();
    });

    // AM/PM controls
    amBtn?.addEventListener('click', (event) => {
        event.stopPropagation();

        currentAmPm = 'AM';

        renderClockDisplay();
        commitClockToInput();
    });

    pmBtn?.addEventListener('click', (event) => {
        event.stopPropagation();

        currentAmPm = 'PM';

        renderClockDisplay();
        commitClockToInput();
    });

    // Set the clock to the current time
    clockNowBtn?.addEventListener('click', (event) => {
        event.stopPropagation();

        const currentTime = new Date();
        const hour = currentTime.getHours();

        currentMin = currentTime.getMinutes();
        currentAmPm = hour >= 12 ? 'PM' : 'AM';
        currentHour12 = hour % 12;

        if (currentHour12 === 0) {
            currentHour12 = 12;
        }

        renderClockDisplay();
        commitClockToInput();
    });

    // Preset time buttons
    document.querySelectorAll('.preset-chip').forEach((chip) => {
        chip.addEventListener('click', (event) => {
            event.stopPropagation();

            const timePreset = chip.getAttribute('data-time');

            if (timePreset && timeInput) {
                timeInput.value = timePreset;
                loadTimeIntoClock();
                commitClockToInput();
            }
        });
    });

    // Confirm clock selection
    clockDoneBtn?.addEventListener('click', (event) => {
        event.stopPropagation();

        commitClockToInput();
        closeClock();
    });

    // Close clock when clicking outside
    document.addEventListener('click', (event) => {
        if (!clockDropdown || clockDropdown.hasAttribute('hidden')) {
            return;
        }

        if (
            !clockDropdown.contains(event.target) &&
            event.target !== timeInput &&
            event.target !== timePickerBtn
        ) {
            closeClock();
        }
    });

    // Close clock using Escape
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            closeClock();
        }
    });

    // -------------------------------------------------------------
    // 5. Populate Categories Dropdown
    // -------------------------------------------------------------
    const categorySelect = document.getElementById('category');

    if (categorySelect) {
        const categories = [
            { id: 'entertainment', name: 'Entertainment' },
            { id: 'food_dining', name: 'Food & Dining' },
            { id: 'gas_transport', name: 'Gas & Transport' },
            { id: 'grocery_net', name: 'Grocery (Online)' },
            { id: 'grocery_pos', name: 'Grocery (In-Store / POS)' },
            { id: 'health_fitness', name: 'Health & Fitness' },
            { id: 'home', name: 'Home' },
            { id: 'kids_pets', name: 'Kids & Pets' },
            { id: 'misc_net', name: 'Miscellaneous (Online)' },
            { id: 'misc_pos', name: 'Miscellaneous (POS)' },
            { id: 'personal_care', name: 'Personal Care' },
            { id: 'shopping_net', name: 'Shopping (Online)' },
            { id: 'shopping_pos', name: 'Shopping (In-Store / POS)' },
            { id: 'travel', name: 'Travel' }
        ];

        categorySelect.innerHTML = categories
            .map((category, index) =>
                `<option value="${category.id}" ${index === 1 ? 'selected' : ''}>${category.name}</option>`
            )
            .join('');
    }

    // -------------------------------------------------------------
    // 6. Submit Form and Display Prediction
    // -------------------------------------------------------------
    transactionForm?.addEventListener('submit', async (event) => {
        // Prevent navigation to the /submit route.
        event.preventDefault();

        if (resultSection) resultSection.hidden = true;
        if (errorElement) errorElement.hidden = true;

        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = 'Analyzing...';
        }

        try {
            syncDateTime();

            const response = await fetch('/submit', {
                method: 'POST',
                body: new FormData(transactionForm)
            });

            // Safely handle responses that are not valid JSON.
            let result;

            try {
                result = await response.json();
            } catch {
                throw new Error('The server returned an invalid response. Check your FastAPI backend.');
            }

            if (!response.ok) {
                const detail = result.detail;

                const message = Array.isArray(detail)
                    ? detail.map((item) =>
                        `${item.loc?.join('.') || 'Form'}: ${item.msg}`
                    ).join('; ')
                    : detail || 'Prediction failed';

                throw new Error(message);
            }

            // Display the prediction returned by FastAPI.
            const prediction = result.prediction;
            const probability = result.fraud_probability;

            const resultTitle = document.getElementById('result-title');
            const riskPercent = document.getElementById('risk-percent');
            const riskLevel = document.getElementById('risk-level');
            const notice = document.getElementById('notice');

            if (resultTitle) {
                resultTitle.textContent = 'Transaction Analysis Result';
            }

            if (riskPercent) {
                if (probability !== undefined && probability !== null) {
                    riskPercent.textContent =
                        `${(Number(probability) * 100).toFixed(2)}%`;
                    riskLevel.textContent = 'Estimated fraud probability';
                } else {
                    const isFraud = Number(prediction) === 1;

                    riskPercent.textContent = isFraud
                        ? 'Fraud detected'
                        : 'No fraud detected';

                    riskLevel.textContent = `Predicted class: ${prediction}`;
                }
            }

            if (riskLevel) {
                riskLevel.textContent =
                    probability !== undefined && probability !== null
                        ? 'Estimated fraud probability'
                        : 'Model prediction class';
            }

            if (notice) {
                notice.textContent = 'Prediction generated successfully.';
            }

            if (resultSection) {
                resultSection.hidden = false;
                resultSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'nearest'
                });
            }

            if (prediction === undefined && probability === undefined) {
                throw new Error(
                    'The server response does not contain "prediction" or "fraud_probability". Check the /submit route.'
                );
            }

        } catch (error) {
            if (errorElement) {
                errorElement.textContent = error.message;
                errorElement.hidden = false;
            }

            console.error('Transaction analysis failed:', error);

        } finally {
            if (submitButton) {
                submitButton.disabled = false;
                submitButton.textContent = 'Analyze transaction';
            }
        }
    });
});