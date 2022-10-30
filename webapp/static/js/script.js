var options = [];
var defapair = defpair;
var js_strings = JSON.parse(productsjsondata);
var js_approved_strings = JSON.parse(aproductsjsondata)
var listLength = js_strings.length;

options.push('<option value="None">None</option>');
for (var i = 0; i < listLength; i++){
    if (js_strings[i] === defapair) {
        options.push('<option value="' + js_strings[i] + '" selected>' + js_strings[i] + '</option>');
    } else { if (js_approved_strings.includes(js_strings[i])) {
        options.push('<option value="' + js_strings[i] + '">' + js_strings[i] + '</option>');
    }
    }   
}

$('<select/>', {
    'class': 'single',
    'name': 'currency-pairs',
    'id': 'currency-pairs',
    html: options.join('')
}).appendTo('form');