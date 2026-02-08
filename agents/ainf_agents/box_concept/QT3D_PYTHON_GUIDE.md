# Guía para Insertar Objetos en Qt3D con Python (PySide6)

## Lecciones Aprendidas

Este documento resume las lecciones aprendidas tras múltiples intentos para renderizar objetos en Qt3D usando Python/PySide6.

---

## ✅ Patrón que FUNCIONA

### 1. Estructura Básica de un Objeto

```python
# Crear entity como hijo del root_entity o de otro entity padre
entity = Qt3DCore.QEntity(parent_entity)

# Crear mesh
mesh = Qt3DExtras.QCuboidMesh()  # o QSphereMesh, QCylinderMesh, etc.
mesh.setXExtent(0.5)
mesh.setYExtent(0.5)
mesh.setZExtent(0.5)

# Crear material - NUEVO para cada objeto
material = Qt3DExtras.QPhongMaterial()
material.setDiffuse(QColor(255, 180, 0))
material.setAmbient(QColor(255, 180, 0).darker(120))

# Crear transform
transform = Qt3DCore.QTransform()
transform.setTranslation(QVector3D(x, y, z))

# Añadir componentes al entity
entity.addComponent(mesh)
entity.addComponent(material)
entity.addComponent(transform)
```

### 2. Regla CRÍTICA: Mantener Referencias

**El garbage collector de Python eliminará los objetos si no mantienes referencias a ellos.**

```python
# ❌ INCORRECTO - Los objetos pueden ser eliminados por el GC
def create_object(parent):
    entity = Qt3DCore.QEntity(parent)
    mesh = Qt3DExtras.QCuboidMesh()
    # ... etc
    return entity  # Solo devuelve entity, mesh/material/transform se pierden

# ✅ CORRECTO - Mantener TODAS las referencias
def create_object(parent):
    entity = Qt3DCore.QEntity(parent)
    mesh = Qt3DExtras.QCuboidMesh()
    material = Qt3DExtras.QPhongMaterial()
    transform = Qt3DCore.QTransform()
    
    # ... configurar y añadir componentes ...
    
    # Devolver TODAS las referencias
    return {
        'entity': entity,
        'mesh': mesh,
        'material': material,
        'transform': transform
    }
```

### 3. Almacenar en Estructuras Persistentes

```python
class MyVisualizer:
    def __init__(self):
        self.scene_objects = {}  # Diccionario para mantener referencias
    
    def create_object(self, obj_id, parent):
        entity = Qt3DCore.QEntity(parent)
        mesh = Qt3DExtras.QCuboidMesh()
        material = Qt3DExtras.QPhongMaterial()
        transform = Qt3DCore.QTransform()
        
        # Configurar...
        entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)
        
        # GUARDAR todas las referencias
        self.scene_objects[f'object_{obj_id}'] = {
            'entity': entity,
            'mesh': mesh,
            'material': material,
            'transform': transform
        }
```

---

## 🔧 Crear Objetos Compuestos (con hijos)

Para objetos complejos como mesas, sillas, o wireframes:

```python
def create_table(self, table_id, cx, cy):
    # 1. Entity padre con su transform global
    table_entity = Qt3DCore.QEntity(self.root_entity)
    
    table_transform = Qt3DCore.QTransform()
    table_transform.setTranslation(QVector3D(cx, cy, 0))
    table_entity.addComponent(table_transform)
    
    # 2. Crear materiales (pueden compartirse o ser únicos)
    top_material = Qt3DExtras.QPhongMaterial()
    top_material.setDiffuse(QColor(139, 90, 43))
    
    # 3. Crear partes como hijos del entity padre
    # TABLERO
    top_mesh = Qt3DExtras.QCuboidMesh()
    top_mesh.setXExtent(width)
    top_mesh.setYExtent(depth)
    top_mesh.setZExtent(0.03)
    
    top_entity = Qt3DCore.QEntity(table_entity)  # Hijo de table_entity
    top_transform = Qt3DCore.QTransform()
    top_transform.setTranslation(QVector3D(0, 0, height))
    
    top_entity.addComponent(top_mesh)
    top_entity.addComponent(top_transform)
    top_entity.addComponent(top_material)
    
    # 4. GUARDAR TODAS las referencias
    self.scene_objects[f'table_{table_id}'] = {
        'entity': table_entity,
        'transform': table_transform,
        'top_entity': top_entity,
        'top_mesh': top_mesh,
        'top_transform': top_transform,
        'top_material': top_material,
    }
```

---

## 📦 Crear Wireframe (Caja de Líneas)

Para un wireframe, usar cuboids delgados es más simple que cilindros con rotaciones:

```python
def create_wireframe_box(self, obj_id, cx, cy, cz, width, depth, height, color):
    # Entity padre
    box_entity = Qt3DCore.QEntity(self.root_entity)
    box_transform = Qt3DCore.QTransform()
    box_transform.setTranslation(QVector3D(cx, cy, cz))
    box_entity.addComponent(box_transform)
    
    line_thickness = 0.02
    hw, hd = width / 2, depth / 2
    
    # Almacenar referencias
    edges_data = []
    
    # Definir los 12 edges del cubo
    # 4 verticales
    for x, y in [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]:
        edges_data.append((x, y, height/2, line_thickness, line_thickness, height))
    
    # 4 horizontales abajo (z=0)
    for y in [-hd, hd]:
        edges_data.append((0, y, line_thickness/2, width, line_thickness, line_thickness))
    for x in [-hw, hw]:
        edges_data.append((x, 0, line_thickness/2, line_thickness, depth, line_thickness))
    
    # 4 horizontales arriba (z=height)
    for y in [-hd, hd]:
        edges_data.append((0, y, height-line_thickness/2, width, line_thickness, line_thickness))
    for x in [-hw, hw]:
        edges_data.append((x, 0, height-line_thickness/2, line_thickness, depth, line_thickness))
    
    edge_refs = []
    for ex, ey, ez, sx, sy, sz in edges_data:
        edge_entity = Qt3DCore.QEntity(box_entity)
        
        mesh = Qt3DExtras.QCuboidMesh()
        mesh.setXExtent(sx)
        mesh.setYExtent(sy)
        mesh.setZExtent(sz)
        
        material = Qt3DExtras.QPhongMaterial()
        material.setDiffuse(color)
        material.setAmbient(color.darker(120))
        
        transform = Qt3DCore.QTransform()
        transform.setTranslation(QVector3D(ex, ey, ez))
        
        edge_entity.addComponent(mesh)
        edge_entity.addComponent(material)
        edge_entity.addComponent(transform)
        
        # GUARDAR referencias de cada edge
        edge_refs.append({
            'entity': edge_entity,
            'mesh': mesh,
            'material': material,
            'transform': transform
        })
    
    # GUARDAR todo
    self.scene_objects[f'wireframe_{obj_id}'] = {
        'entity': box_entity,
        'transform': box_transform,
        'edges': edge_refs
    }
```

---

## ⚠️ Errores Comunes

### 1. No mantener referencias
```python
# ❌ MAL - Variables locales se pierden
mesh = Qt3DExtras.QCuboidMesh()
entity.addComponent(mesh)
# mesh se elimina al salir del scope

# ✅ BIEN - Guardar en estructura persistente
self.meshes.append(mesh)
```

### 2. Reutilizar el mismo material para múltiples entities
```python
# ❌ PUEDE causar problemas
material = Qt3DExtras.QPhongMaterial()
for entity in entities:
    entity.addComponent(material)  # El mismo material

# ✅ MEJOR - Material nuevo para cada entity
for entity in entities:
    material = Qt3DExtras.QPhongMaterial()
    material.setDiffuse(color)
    entity.addComponent(material)
```

### 3. Olvidar añadir el transform al entity padre
```python
# ❌ MAL - Entity padre sin transform
parent_entity = Qt3DCore.QEntity(root)
# Falta: parent_entity.addComponent(transform)

# ✅ BIEN
parent_entity = Qt3DCore.QEntity(root)
parent_transform = Qt3DCore.QTransform()
parent_transform.setTranslation(QVector3D(x, y, z))
parent_entity.addComponent(parent_transform)
```

---

## 🗑️ Eliminar Objetos

```python
def remove_object(self, obj_id):
    key = f'object_{obj_id}'
    if key in self.scene_objects:
        # Desconectar del árbol de entities
        self.scene_objects[key]['entity'].setParent(None)
        # Eliminar referencia
        del self.scene_objects[key]
```

---

## 📋 Checklist para Depuración

Si un objeto no se ve:

1. ☐ ¿El entity es hijo de root_entity o de otro entity visible?
2. ☐ ¿Se añadieron los 3 componentes: mesh, material, transform?
3. ☐ ¿Se mantienen referencias a TODOS los objetos (entity, mesh, material, transform)?
4. ☐ ¿El transform tiene una posición visible en la escena?
5. ☐ ¿El material tiene colores configurados (diffuse, ambient)?
6. ☐ ¿El mesh tiene dimensiones > 0?
7. ☐ ¿Hay iluminación en la escena?

---

## 🎯 Resumen

**La regla de oro**: En Python con Qt3D, SIEMPRE mantén referencias a TODOS los objetos Qt3D que crees (entities, meshes, materials, transforms). El garbage collector de Python es agresivo y eliminará cualquier objeto sin referencias, incluso si está conectado al árbol de entities de Qt3D.
