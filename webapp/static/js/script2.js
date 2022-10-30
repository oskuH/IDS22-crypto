var options = [];
var defatime = deftime
var js_times = ["1 minute", "5 minutes", "15 minutes", "1 hour", "6 hours", "24 hours"]
var listLength = js_times.length;

for (var i = 0; i < listLength; i++){
    if (js_times[i] === defatime) {
        options.push('<option value="' + js_times[i] + '" selected>' + js_times[i] + '</option>');
    } else {
        options.push('<option value="' + js_times[i] + '">' + js_times[i] + '</option>');
    }   
}

$('<select/>', {
    'class': 'single',
    'name': 'times',
    'id': 'times',
    html: options.join('')
}).appendTo('form');