import os
import sqlite3

from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash, get_flashed_messages, jsonify
from datetime import datetime
from werkzeug.utils import secure_filename
 

# Telling application flask everything you need to run application is in same directory
app = Flask(__name__)



#==========  Configuration   ==========# 
#! we can use this to ensure that only boards that has api key of system who could communicate with server
#! we can have seprate data base of multiple of api keys 
app.secret_key = 'your_secret_key'  

UPLOAD_FOLDER = 'firmware'
ALLOWED_EXTENSIONS = {'bin'}
DATABASE = 'devices.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB
#======================================#


# Ensure the upload folder exists if not exist it create it new one 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def init_db():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        # Existing tables
        c.execute('''
            CREATE TABLE IF NOT EXISTS devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT UNIQUE NOT NULL,
                last_seen TIMESTAMP NOT NULL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS firmware (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                filename TEXT NOT NULL,
                upload_date TIMESTAMP NOT NULL
            )
        ''')
        # New mapping table
        c.execute('''
            CREATE TABLE IF NOT EXISTS device_firmware (
                device_id TEXT NOT NULL,
                firmware_id INTEGER NOT NULL,
                FOREIGN KEY(device_id) REFERENCES devices(device_id),
                FOREIGN KEY(firmware_id) REFERENCES firmware(id),
                PRIMARY KEY(device_id)
            )
        ''')
        conn.commit()


#==========  Initialize the database   ==========# 
init_db()
#================================================#


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/devices')
def devices():
    with sqlite3.connect(DATABASE) as conn:
        
        conn.row_factory = sqlite3.Row  # Access columns by name

        c = conn.cursor()
        
        c.execute('''
            SELECT devices.device_id, devices.last_seen, firmware.version AS firmware_version
            FROM devices
            LEFT JOIN device_firmware ON devices.device_id = device_firmware.device_id
            LEFT JOIN firmware ON device_firmware.firmware_id = firmware.id
        ''')
        
        devices = c.fetchall()
    return render_template('devices.html', devices=devices)



def allowed_file(filename):

    if ( '.' not in filename ):
        return False

    file_extension = filename.rsplit('.', 1)[1].lower()

    if ( file_extension in ALLOWED_EXTENSIONS ):
        return True

    else:
        return False


@app.before_request
def validate_upload_size():

    if (request.content_length > MAX_FILE_SIZE):

        flash('File is too large') # used to send a one-time message from the backend to the frontend
        return redirect(url_for('upload_firmware'))

    




@app.route('/upload', methods=['GET', 'POST'])
def upload_firmware():
    if (request.method == 'POST'):
        version = request.form['version']
        device_id = request.form.get('device_id')  # May be None for global firmware
        file = request.files['file']


        if (file and allowed_file(file.filename) ):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            with sqlite3.connect(DATABASE) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO firmware (version, filename, upload_date) VALUES (?, ?, ?)",
                          (version, filename, datetime.utcnow()))
                firmware_id = c.lastrowid  # Get the inserted firmware's ID

                if (device_id):
                    # Assign the firmware to the device
                    c.execute('''
                        INSERT OR REPLACE INTO device_firmware (device_id, firmware_id)
                        VALUES (?, ?)
                    ''', (device_id, firmware_id))
                conn.commit()
            flash('Firmware uploaded successfully!')
            return redirect(url_for('index'))
        else:
            flash('Invalid file type. Only .bin files are allowed.')
    else:
        # Fetch the list of devices
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute('SELECT device_id FROM devices')
            devices = c.fetchall()
        return render_template('upload.html', devices=devices)


@app.route('/api/check_update', methods=['POST'])
def check_update():
    data = request.get_json()
    device_id = data.get('device_id')
    current_version = data.get('current_version')

    # Update or insert the device's last seen time
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO devices (device_id, last_seen) VALUES (?, ?)",
                  (device_id, datetime.utcnow()))
        conn.commit()

        # Check for device-specific firmware
        c.execute('''
            SELECT firmware.version, firmware.filename
            FROM device_firmware
            JOIN firmware ON device_firmware.firmware_id = firmware.id
            WHERE device_firmware.device_id = ?
        ''', (device_id,))
        firmware = c.fetchone()

        if ( firmware and firmware[0] != current_version ):

            # New device-specific firmware available
            response = {
                'update_available': True,
                'version': firmware[0],
                'url': request.host_url + 'firmware/' + firmware[1]
            }
            return jsonify(response)

        # Check for global firmware
        c.execute('''
            SELECT version, filename
            FROM firmware
            WHERE id NOT IN (SELECT firmware_id FROM device_firmware)
            ORDER BY upload_date DESC LIMIT 1
        ''')
        global_firmware = c.fetchone()

        if ( global_firmware and global_firmware[0] != current_version ):
            # New global firmware available
            response = {
                'update_available': True,
                'version': global_firmware[0],
                'url': request.host_url + 'firmware/' + global_firmware[1]
            }
            return jsonify(response)
        else:
            # No update available
            return jsonify({'update_available': False})


@app.route('/firmware/<filename>')
def download_firmware(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/assign_firmware/<device_id>', methods=['GET', 'POST'])
def assign_firmware(device_id):
    if ( request.method == 'POST' ):
        firmware_id = request.form['firmware_id']

        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            # Assign the firmware to the device
            c.execute('''
                INSERT OR REPLACE INTO device_firmware (device_id, firmware_id)
                VALUES (?, ?)
            ''', (device_id, firmware_id))
            conn.commit()
        flash('Firmware assigned successfully!')
        return redirect(url_for('devices'))
    else:
        # Fetch available firmware versions
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute('SELECT id, version FROM firmware ORDER BY upload_date DESC')
            firmware_list = c.fetchall()
        return render_template('assign_firmware.html', device_id=device_id, firmware_list=firmware_list)


#======   Here we can adjust the display based on the HTTP code   ======# 
#=======================================================================#
   


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
