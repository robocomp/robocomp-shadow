var boxes = [];
var selectedId = null;
var found = false;
var clickedBox = null;

// Obtener los bounding boxes del servidor cada segundo
setInterval(function() {
    $.get('/get_boxes', function(data) {
        boxes = data;

        // Comprobar si el ID seleccionado sigue apareciendo en los nuevos boxes
        if (selectedId !== null) {
            found = false;
            for (var i = 0; i < boxes.length; i++) {
                if (boxes[i].id === selectedId) {
                    found = true;
                    clickedBox = boxes[i];
                    break;
                }
            }
            if (!found) {
                console.log('El ID seleccionado ya no aparece en los nuevos bounding boxes.');
                selectedId = null;
                clickedBox = null;
                document.getElementById('indicator').style.backgroundColor = 'red';
                document.getElementById('indicator-text').textContent = 'No seleccionado';
            }
        }

        // Verificar si clickedBox tiene valor
        if (clickedBox !== null) {
            document.getElementById('indicator').style.backgroundColor = 'green';
            document.getElementById('indicator-text').textContent = 'Seleccionado';

            // Enviar el bounding box seleccionado al servidor
            $.ajax({
                url: '/select_person',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({'box': clickedBox}),
                success: function(data) {
                    if (data.status === 'success') {
                        // Remover la clase "selected-box" de todos los bounding boxes
                        $('.bounding-box').removeClass('selected-box');

                        // Agregar la clase "selected-box" al bounding box seleccionado
                        $('.bounding-box').eq(clickedIndex).addClass('selected-box');

                        // Hacer algo cuando la selección es exitosa
                        console.log('Bounding box seleccionado:', clickedBox);
                        alert('Se ha seleccionado un bounding box');
                    }
                }
            });
        } else {
            $.ajax({
                url: '/select_person',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({'box': {}}),
                success: function(data) {
                    if (data.status === 'success') {
                        // Remover la clase "selected-box" de todos los bounding boxes
                        $('.bounding-box').removeClass('selected-box');

                        // Agregar la clase "selected-box" al bounding box seleccionado
                        $('.bounding-box').eq(clickedIndex).addClass('selected-box');

                        // Hacer algo cuando la selección es exitosa
                        console.log('Bounding box seleccionado:', clickedBox);
                        alert('Se ha seleccionado un bounding box');
                    }
                }
            });
        }
    });
}, 1000);

// Manejar el evento de clic
// Manejar el evento de clic
$('#video').click(function(e) {
    // Calcular la posición del elemento
    var elementPosition = $(this).offset();

    // Obtener las coordenadas del clic relativas al elemento de la imagen
    var mouseX = e.pageX - elementPosition.left;
    var mouseY = e.pageY - elementPosition.top;

    // Determinar en qué bounding box se hizo clic
    clickedBox = null;
    var clickedIndex = null;
    for (var i = 0; i < boxes.length; i++) {
        var box = boxes[i];
        if (mouseX >= box.x && mouseX <= box.x + box.width &&
            mouseY >= box.y && mouseY <= box.y + box.height) {
            clickedBox = box;
            clickedIndex = i;
            selectedId = box.id;
            break;
        }
    }

    if (clickedBox === null) {
        selectedId = null;
        document.getElementById('indicator').style.backgroundColor = 'red';
        document.getElementById('indicator-text').textContent = 'No seleccionado';
    }
});

