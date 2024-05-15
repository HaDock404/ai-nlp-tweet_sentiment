function display_cloudWords() {
    var choice1 = document.querySelector('input[name="image_choice1"]:checked').value;
    var choice2 = document.querySelector('input[name="image_choice2"]:checked').value;

    var displayed_image_1 = document.getElementById('main-paragraph-image-display_1');
    var displayed_image_2 = document.getElementById('main-paragraph-image-display_2');

    switch (choice1) {
        case 'image1':
            displayed_image_1.src = "../static/images/df_pos.png";
            break;
        case 'image2':
            displayed_image_1.src = "../static/images/df_neg.png";
            break;
        case 'image3':
            displayed_image_1.src = "../static/images/df_pos.png";
            break;
        default:
            displayed_image_1.src = "../static/images/df_pos.png";
    }

    switch (choice2) {
        case 'image3':
            displayed_image_2.src = "../static/images/count_pos.png";
            break;
        case 'image4':
            displayed_image_2.src = "../static/images/count_neg.png";
            break;
        default:
            displayed_image_2.src = "../static/images/count_pos.png";
    }
}

document.getElementById('submitBtn').addEventListener('click', function() {
    var selectedOption = document.getElementById('options').value;
    console.log('Option sélectionnée :', selectedOption);
  });




console.log("Le fichier JavaScript est correctement chargé.");