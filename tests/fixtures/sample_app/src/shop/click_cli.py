import click


@click.group()
def admin():
    """Admin commands."""


@admin.command()
@click.option("--count", "-c", type=int, default=1, help="How many")
@click.option("--dry-run", is_flag=True)
@click.argument("name")
def seed_data(count, dry_run, name):
    """Seed the database."""


@click.command(name="standalone")
def lone():
    pass
