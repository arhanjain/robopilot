import network_as_code as nac
import time
 
from network_as_code.models.device import Device, DeviceIpv4Addr
 
from network_as_code.models.slice import (
    Point,
    AreaOfService,
    NetworkIdentifier,
    SliceInfo,
    Throughput,
    TrafficCategories,
    Apps
)

class NetworkSlicer:
    def wait_for(self, slice, desired_state):
        while slice.state != desired_state:
            time.sleep(1)
            slice.refresh()
            print(slice.state)
    def __init__(self) -> None:
        client = nac.NetworkAsCodeClient(
            token="33d8e864d9msh2c4370098b7bffep1ceae9jsnab9fe827fcda"
        )

        device = client.devices.get(
                "bad5a2f1-496f-43fb-b438-5f372bfd32a0@testcsp.net",
                ipv4_address=DeviceIpv4Addr(
                    public_address="233.252.0.2",
                    private_address="192.0.2.25",
                    public_port=80
                    )
                )

        
        # Creation of a slice:
        # We use the country code (MCC) and network code (MNC) to identify the network
        # Different types of slices can be requested using service type and differentiator
        # The area of the slice must also be described in geo-coordinates
        my_slice = client.slices.create(
            name="slice-name",
            network_id = NetworkIdentifier(mcc="236", mnc="30"),
            slice_info = SliceInfo(service_type="eMBB", differentiator="123456"),
            area_of_service=AreaOfService(polygon=[
                Point(latitude=42.0, longitude=42.0),
                Point(latitude=41.0, longitude=42.0),
                Point(latitude=42.0, longitude=41.0),
                Point(latitude=42.0, longitude=42.0)
            ]),
            # Use HTTPS to send notifications
            notification_url="http://notify.me/here",
            notification_auth_token="replace-with-your-auth-token"
        )
        print("requested")
        self.wait_for(my_slice, "AVAILABLE")
        print("available")

        my_slice.activate()
        self.wait_for(my_slice, "OPERATING")
        print("operating")

        #attaching device:
        my_slice.attach(
            device,
            traffic_categories=TrafficCategories(
                apps=Apps(
                    os="97a498e3-fc92-5c94-8986-0333d06e4e47",
                    apps=["ENTERPRISE", "ENTERPRISE2"]
                )
            ),
            notification_url="https://notify.me/here",
            notification_auth_token="replace-with-your-auth-token"
        )

        self.slice = my_slice

    def close(self):
        my_slice = self.slice
        my_slice.deactivate()
        self.wait_for(my_slice, "AVAILABLE")
        print("deactivate")
        my_slice.delete()
        print("deleted")


t = NetworkSlicer()
t.close()
