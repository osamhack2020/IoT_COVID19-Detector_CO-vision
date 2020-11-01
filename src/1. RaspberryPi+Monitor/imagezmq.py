import zmq
import numpy as np


class ImageSender():
    def __init__(self, connect_to='tcp://127.0.0.1:5555'):

        self.zmq_context = SerializingContext()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect(connect_to)

    def send_image(self, msg, image):

        if image.flags['C_CONTIGUOUS']:
            self.zmq_socket.send_array(image, msg, copy=False)
        else:
            image = np.ascontiguousarray(image)
            self.zmq_socket.send_array(image, msg, copy=False)
        hub_reply = self.zmq_socket.recv()  # receive the reply message
        return hub_reply

    def send_jpg(self, msg, jpg_buffer):

        self.zmq_socket.send_jpg(msg, jpg_buffer, copy=False)
        hub_reply = self.zmq_socket.recv()  # receive the reply message
        return hub_reply


class ImageHub():
    def __init__(self, open_port='tcp://*:5555'):

        self.zmq_context = SerializingContext()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind(open_port)

    def recv_image(self, copy=False):
        msg, image = self.zmq_socket.recv_array(copy=False)
        return msg, image

    def recv_jpg(self, copy=False):

        msg, jpg_buffer = self.zmq_socket.recv_jpg(copy=False)
        return msg, jpg_buffer

    def send_reply(self, reply_message=b'OK'):
        self.zmq_socket.send(reply_message)


class SerializingSocket(zmq.Socket):

    def send_array(self, A, msg='NoName', flags=0, copy=True, track=False):
        md = dict(
            msg=msg,
            dtype=str(A.dtype),
            shape=A.shape,
        )
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(A, flags, copy=copy, track=track)

    def send_jpg(self,
                 msg='NoName',
                 jpg_buffer=b'00',
                 flags=0,
                 copy=True,
                 track=False):
        md = dict(msg=msg, )
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(jpg_buffer, flags, copy=copy, track=track)

    def recv_array(self, flags=0, copy=True, track=False):
        md = self.recv_json(flags=flags)
        msg = self.recv(flags=flags, copy=copy, track=track)
        A = np.frombuffer(msg, dtype=md['dtype'])
        return (md['msg'], A.reshape(md['shape']))

    def recv_jpg(self, flags=0, copy=True, track=False):
        md = self.recv_json(flags=flags)  # metadata text
        jpg_buffer = self.recv(flags=flags, copy=copy, track=track)
        return (md['msg'], jpg_buffer)

class SerializingContext(zmq.Context):
    _socket_class = SerializingSocket