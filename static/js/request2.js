$(function() {
    $('#signup2').click(function() {
 
        $.ajax({
            url: '/ans2',
            data: $('form').serialize(),
            type: 'POST',
            success: function(response) {
                console.log(response);
				window.location = "answer.html";
            },
            error: function(error) {
                console.log(error);
            }
        });
    });
});