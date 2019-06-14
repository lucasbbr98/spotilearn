(function poll() {
    setTimeout(function () {
        $.ajax({
            type: 'GET',
            url: '/loading_status',
            success: function (data) {
                console.log(data)
            },
            complete: poll
        });
    }, 5000);
})();