from ebaysdk.trading import Connection as Trading
from ebaysdk.exception import ConnectionError

from settings import PROJECT_ROOT, PAYPAL_EMAIL, COUNTRY, POSTAL_CODE

from .fetch_data import fetch_card_price


class CardListingObject:
    def __init__(self, path_to_card_image, card):
        self.path_to_card_image = path_to_card_image
        self.card_id = card['id']
        self.card_name = card['name']
        self.card_set = card['set']
        self.title = self._get_title(self.card_name, self.card_set)

        self._price = self.set_price()
        self._api = None
        self._image_url = None

    @property
    def item_payload(self):
        return {
            "Item": {
                "Title": self.title,
                "Description": f"Each auction is for 1 copy of shown card.  The card you will receive is displayed in"
                               f" the image.",
                "PrimaryCategory": {
                    "CategoryID": "183454"
                },
                "StartPrice": f"{self._price}",
                "CategoryMappingAllowed": "true",
                "Country": f"{COUNTRY}",
                "ConditionID": "3000",
                "Currency": "USD",
                "DispatchTimeMax": "3",
                "ListingDuration": "Days_7",
                "ListingType": "Chinese",
                "PaymentMethods": "PayPal",
                "PayPalEmailAddress": f"{PAYPAL_EMAIL}",
                "PictureDetails": {
                    "PictureURL": self._image_url
                },
                "PostalCode": f"{POSTAL_CODE}",
                "Quantity": "1",
                "ReturnPolicy": {
                    "ReturnsAcceptedOption": "ReturnsAccepted",
                    "RefundOption": "MoneyBack",
                    "ReturnsWithinOption": "Days_30",
                    "ShippingCostPaidByOption": "Buyer"
                },
                "ShippingDetails": {
                    "ShippingType": "Flat",
                    "ShippingServiceOptions": {
                        "ShippingServicePriority": "1",
                        "ShippingService": "USPSMedia",
                        "ShippingServiceCost": "2.50"
                    }
                },
                "Site": f"{COUNTRY}"
            }
        }

    @property
    def api(self):
        return self._api

    @property
    def image_url(self):
        return self._image_url

    @property
    def price(self):
        return self._price

    @staticmethod
    def _get_title(card_name, card_set):
        return f"MTG [{card_set}]{card_name} x 1"

    def set_price(self):
        self._price = fetch_card_price(self.card_id)
        return self._price

    def activate_api(self):
        """Required for ebay -> app id, dev id, cert id, and token in ebay.yaml:
        For more detailed information on these items, please visit
        'ebaysdk' on github.
        """

        self._api = Trading(config_file=f'{PROJECT_ROOT}/ebay.yaml', domain='api.sandbox.ebay.com')
        return self._api

    def upload_image(self):
        """Uploads to ebay server for use in listing."""

        files = {'file': ('EbayImage', open(self.path_to_card_image, 'rb'))}
        picture_data = {
            "WarningLevel": "Low",
            "PictureName": self.title,
        }
        response = self.api.execute('UploadSiteHostedPictures', picture_data, files=files)
        self._image_url = response.reply.get('SiteHostedPictureDetails').get('FullURL')
        return self._image_url

    def create_listing(self):
        """Requires image to be uploaded."""

        try:
            if self.image_url:
                response = self.api.execute('AddItem', self.item_payload)
                return response.reply
            raise Exception('Must upload image before creating listing.')
        except ConnectionError as e:
            print(e)
            print(e.response.dict())

    def perform_create(self):
        self.activate_api()
        self.upload_image()
        self.create_listing()
