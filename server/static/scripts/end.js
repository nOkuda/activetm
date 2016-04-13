$(document).ready(function() {
  if (!document.hidden) {
    $.ajax({
      type: "GET",
      url: "/finalize",
      headers: {'uuid': Cookies.get('user_study_uuid')},
      success: function(data) {
        correct = data['correct'];
        complete = data['complete'];
        $("#correct").text(correct)
        $("#completed").text(complete)
        percentage = 0;
        if (complete > 0) {
          percentage = 100 * correct / complete
        }
        $("#finalscore").text(percentage + "%")
      }
    });
    Cookies.remove('user_study_uuid');
    Cookies.remove('user_study_num_docs');
  }
});
