# Seguridad

## Cómo reportar un problema

Si encuentras una vulnerabilidad, repórtala de forma privada desde la pestaña Security de este repositorio, con la opción "Report a vulnerability". Por favor no abras un issue público con los detalles.

## Alcance

Es un proyecto académico que corre en la computadora de quien lo usa. No tiene servidor, cuentas de usuario ni datos personales.

- Usa solo datos públicos (UCI Statlog German Credit) y datos simulados que el propio código declara como simulados. No contiene información de ningún banco ni de sus clientes.
- El único paso que usa internet descarga la base original por https, verifica su suma SHA-256 y extrae solo el archivo esperado.
- Las dependencias están fijadas a una versión y Dependabot las revisa cada semana. Las acciones de GitHub están fijadas a un commit.
