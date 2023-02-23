# class BankingSystem:
#     def __init__(self):
#         self.accounts = {}
#
#     def create_account(self, timestamp, accountid):
#         if accountid in self.accounts:
#             return "false"
#         self.accounts[accountid] = 0
#         return "true"
#
#     def deposit(self, timestamp, accountid, amount):
#         if accountid not in self.accounts:
#             return ""
#         self.accounts[accountid] += amount
#         return str(self.accounts[accountid])
#
#     def pay(self, timestamp, accountid, amount):
#         if accountid not in self.accounts or self.accounts[accountid] < amount:
#             return ""
#         self.accounts[accountid] -= amount
#         return str(self.accounts[accountid])
#
#
# def solution(queries):
#     bank = BankingSystem()
#     result = []
#     for query in queries:
#         if query[0] == "CREATE ACCOUNT":
#             result.append(bank.create_account(query[1], query[2]))
#         elif query[0] == "DEPOSIT":
#             result.append(bank.deposit(query[1], query[2], query[3]))
#         elif query[0] == "PAY":
#             result.append(bank.pay(query[1], query[2], query[3]))

#     return result
class BankingSystem:
    def _init_(self):
        self.accounts = {}
        self.outgoing_transactions = {}

    def create_account(self, timestamp, account_id):
        if account_id not in self.accounts:
            self.accounts[account_id] = 0
            return "true"
        else:
            return "false"

    def deposit(self, timestamp, account_id, amount):
        if account_id not in self.accounts:
            return ""
        else:
            self.accounts[account_id] += amount
            return str(self.accounts[account_id])

    def pay(self, timestamp, account_id, amount):
        if account_id not in self.accounts or self.accounts[account_id] < amount:
            return ""
        else:
            self.accounts[account_id] -= amount
            if account_id not in self.outgoing_transactions:
                self.outgoing_transactions[account_id] = [timestamp]
            else:
                self.outgoing_transactions[account_id].append(timestamp)
            return str(self.accounts[account_id])

    def rank_accounts(self):
        sorted_outgoing_transactions = sorted(self.outgoing_transactions.items(), key=lambda x: len(x[1]), reverse=True)
        return [x[0] for x in sorted_outgoing_transactions]