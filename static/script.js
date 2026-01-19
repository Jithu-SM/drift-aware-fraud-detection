function toggleTheme() {
    const body = document.body;
    const circle = document.querySelector('.theme-toggle-circle');
    
    if (body.classList.contains('light')) {
        body.classList.remove('light');
        body.classList.add('dark');
        circle.textContent = '🌙';
    } else {
        body.classList.remove('dark');
        body.classList.add('light');
        circle.textContent = '☀️';
    }
}

function toggleReceiverBalance() {
    const transactionType = document.getElementById('transactionType').value;
    const receiverBalanceGroup = document.getElementById('receiverBalanceGroup');
    const receiverInput = receiverBalanceGroup.querySelector('input');
    
    if (transactionType === 'DEBIT' || transactionType === 'CASH_OUT') {
        receiverBalanceGroup.classList.add('hidden');
        receiverInput.removeAttribute('required');
        receiverInput.value = '';
    } else {
        receiverBalanceGroup.classList.remove('hidden');
        receiverInput.setAttribute('required', 'required');
    }
}

// safeguard to ensure DOM is loaded before attaching event listeners
document.addEventListener('DOMContentLoaded', function() {
    toggleReceiverBalance();
});