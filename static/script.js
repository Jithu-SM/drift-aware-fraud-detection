    // Theme toggle functionality
    function toggleTheme() {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    }

    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    // Toggle receiver balance based on transaction type
    function toggleReceiverBalance() {
        const transactionType = document.getElementById('transactionType').value;
        const receiverBalanceGroup = document.getElementById('receiverBalanceGroup');
        const receiverBalanceInput = document.getElementById('oldbalanceDest');
        
        if (transactionType === 'DEBIT' || transactionType === 'CASH_OUT') {
            receiverBalanceGroup.classList.add('hidden');
            receiverBalanceInput.removeAttribute('required');
            receiverBalanceInput.value = '0';
        } else {
            receiverBalanceGroup.classList.remove('hidden');
            receiverBalanceInput.setAttribute('required', 'required');
        }
    }

// safeguard to ensure DOM is loaded before attaching event listeners
document.addEventListener('DOMContentLoaded', function() {
    toggleReceiverBalance();
});