$(function() {
    $(document).on('click','.signup',function() {
 
        $.ajax({
            url: '/ans2',
            data: $('form').serialize(),
            type: 'POST',
            success: function(response) {
                console.log(response);
            },
            error: function(error) {
                console.log(error);
            }
        });
    });
});