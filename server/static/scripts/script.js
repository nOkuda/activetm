// Starrr plugin (https://github.com/dobtco/starrr)
var __slice = [].slice;

(function($, window) {
  var Starrr;

  Starrr = (function() {
    Starrr.prototype.defaults = {
      rating: void 0,
      numStars: 5,
      change: function(e, value) {}
    };

    function Starrr($el, options) {
      var i, _, _ref,
        _this = this;

      this.options = $.extend({}, this.defaults, options);
      this.$el = $el;
      _ref = this.defaults;
      for (i in _ref) {
        _ = _ref[i];
        if (this.$el.data(i) != null) {
          this.options[i] = this.$el.data(i);
        }
      }
      this.createStars();
      this.syncRating();
      this.$el.on('mouseover.starrr', 'span', function(e) {
        return _this.syncRating(_this.$el.find('span').index(e.currentTarget) + 1);
      });
      this.$el.on('mouseout.starrr', function() {
        return _this.syncRating();
      });
      this.$el.on('click.starrr', 'span', function(e) {
        return _this.setRating(_this.$el.find('span').index(e.currentTarget) + 1);
      });
      this.$el.on('starrr:change', this.options.change);
    }

    Starrr.prototype.createStars = function() {
      var _i, _ref, _results;

      _results = [];
      for (_i = 1, _ref = this.options.numStars; 1 <= _ref ? _i <= _ref : _i >= _ref; 1 <= _ref ? _i++ : _i--) {
        _results.push(this.$el.append("<span class='glyphicon .glyphicon-star-empty'></span>"));
      }
      return _results;
    };

    Starrr.prototype.setRating = function(rating) {
// Commented this out to get it to stop freaking out when someone clicked
//   the same rating twice.
//      if (this.options.rating === rating) {
//        rating = void 0;
//      }
// Commented this out to keep the stars from staying filled in after
//   the rating is clicked.
//      this.options.rating = rating;
//      this.syncRating();
      return this.$el.trigger('starrr:change', rating);
    };

    Starrr.prototype.syncRating = function(rating) {
      var i, _i, _j, _ref;

      rating || (rating = this.options.rating);
      if (rating) {
        for (i = _i = 0, _ref = rating - 1; 0 <= _ref ? _i <= _ref : _i >= _ref; i = 0 <= _ref ? ++_i : --_i) {
          this.$el.find('span').eq(i).removeClass('glyphicon-star-empty').addClass('glyphicon-star');
        }
      }
      if (rating && rating < 5) {
        for (i = _j = rating; rating <= 4 ? _j <= 4 : _j >= 4; i = rating <= 4 ? ++_j : --_j) {
          this.$el.find('span').eq(i).removeClass('glyphicon-star').addClass('glyphicon-star-empty');
        }
      }
      if (!rating) {
        return this.$el.find('span').removeClass('glyphicon-star').addClass('glyphicon-star-empty');
      }
    };

    return Starrr;

  })();
  return $.fn.extend({
    starrr: function() {
      var args, option;

      option = arguments[0], args = 2 <= arguments.length ? __slice.call(arguments, 1) : [];
      return this.each(function() {
        var data;

        data = $(this).data('star-rating');
        if (!data) {
          $(this).data('star-rating', (data = new Starrr($(this), option)));
        }
        if (typeof option === 'string') {
          return data[option].apply(data, args);
        }
      });
    }
  });
})(window.jQuery, window);

$(function() {
  return $(".starrr").starrr();
});

$( document ).ready(function() {
  //Set up global variables (terrible, I know)
  var docnumber;
  var d = new Date();
  var starttime = d.getTime();
  var alreadyguessed = false

  //Send data when a star rating is given and then show feedback
  $('#stars').on('starrr:change', function(e, value){
    if (!alreadyguessed) {
      alreadyguessed = true
      d = new Date();
      var endtime = d.getTime();
      var guess = value;
      //Send the data over to the server when the user gives a rating
      $.ajax({
        type: 'POST',
        url: "/rated",
        data: JSON.stringify({
          "rating": value,
          "start_time": starttime,
          "end_time": endtime,
          "uid": Cookies.get('user_study_uuid'),
          "doc_number": docnumber
        }),
        dataType: "json",
        contentType: "application/json",
        success: function(data) {
          //Hide stars when showing feedback
          $("#stars").hide();
          var message = "<button id=continueButton class=\"btn btn-default\">";
          message += "Continue</button>";
          if (guess === data["label"]) {
            message += "<p>You were <span class='correct'>correct</span>.</p>";
          } else {
            message += "<p>You were <span class='incorrect'>incorrect</span>.</p>";
          }
          message += "<p>Your guess: "+guess+"</p>";
          message += "<p>Correct answer: "+data["label"]+"</p>";
          $("#correct").text(data["correct"]);
          $("#progress").text(data["completed"]);
          $("#feedback").html(message);
          $("#continueButton").click(function() {
            alreadyguessed = false
            getDoc();
            $("#feedback").html("");
            $("#stars").show();
          })
        }
      });
    }
  });

  //Function that gets documents
  var getDoc = function() {
    $.ajax({
      type: 'GET',
      url: "/get_doc",
      headers: {'uuid': Cookies.get('user_study_uuid')},
      success: function(data) {
        //If we're done getting documents, redirect to end page
        if (data['doc_number'] === 0) {
          location.href = '/end.html';
          return false;
        }
        $("#document").text(data["document"]);
        docnumber = data["doc_number"];
        d = new Date();
        starttime = d.getTime();
      }
    });
  }

  //Function that gets the user's old document (if they refreshed)
  var getOldDoc = function() {
    $.ajax({
      type: 'GET',
      url: '/old_doc',
      headers: {'uuid': Cookies.get('user_study_uuid')},
      success: function(data) {
        if (data['doc_number'] === 0) {
          location.href = '/end.html';
          return false;
        }
        $("#correct").text(data["correct"]);
        $("#progress").text(data["completed"]);
        $('#document').text(data['document']);
        docnumber = data["doc_number"];
      }
    });
  }

  //Check whether this person is in the middle of the study (has a uuid),
  //  give them one if not (so they start at the beginning of the process)
  if (Cookies.get('user_study_uuid') === undefined) {
    $.get("/uuid", function(data) {
      Cookies.set('user_study_uuid', data.id);
      Cookies.set('user_study_num_docs', 0);
      //Only get a first random document if they are new
      getDoc();
    });
  }
  //If they are not new, get their old document so they can rate it
  else { getOldDoc(); }

});
