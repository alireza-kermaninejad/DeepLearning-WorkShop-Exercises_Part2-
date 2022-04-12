from flask import Blueprints, render_template, request

errors_handlers = Blueprints("errors_handlers", __name__)

@errors_handlers.errorhandler(404)
def error_404(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(400)
def error_400(error):
    return render_template('errors/400.html'), 400

@errors_handlers.errorhandler(403)
def error_403(error):
    return render_template('errors/403.html'), 403

@errors_handlers.errorhandler(500)
def error_500(error):
    return render_template('errors/500.html'), 500

@errors_handlers.errorhandler(415)
def error_415(error):
    return render_template('errors/415.html'), 415